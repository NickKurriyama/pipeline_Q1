"""
run_budget.py  --  Claim C3: bounded per-window compute under a hard active-set budget B.

Unlike demo.py (which only zeroes gradients, saving *updates* but not *compute*), this
experiment realizes the ACTUAL compute saving: per window the non-firing Gaussians are
rendered once into a detached cache, and every inner optimization step renders and
differentiates ONLY the <=B active Gaussians (splat2d.render_active — an exact split for
the additive toy renderer). We then measure real wall-clock ms per window as a function
of the budget B, together with the final reconstruction quality.

Expected shape of the result: optimization cost per window grows with B and saturates;
quality degrades gracefully as B shrinks — i.e. a tunable real-time knob.

Run:  python experiments/run_budget.py
Outputs: outputs/budget.png, outputs/budget.csv
"""
from __future__ import annotations
import os, time
import torch
from common import CFG, SEEDS, OUTDIR, mean_std, fmt, write_csv

from evspikegs.splat2d import (GaussianScene2D, base_grid, render, log_intensity,
                               gaussian_weights, render_inactive_cache, render_active)
from evspikegs.event_data import accumulate_events, make_circular_poses, gt_log_at_pose
from evspikegs.event_loss import event_photometric_loss, anchor_loss
from evspikegs.lif_gate import LIFGate
from evspikegs.metrics import psnr
from evspikegs.flops import budget_window_mflops
from demo import build_gt

BUDGETS = [4, 8, 16, 32, 64, None]        # None = uncapped (dense-style upper bound)
THETA = 0.02                              # low threshold => many fire => the cap binds


def train_budget(budget, seed):
    H, W, n_gauss = CFG["H"], CFG["W"], CFG["n_gauss"]
    n_windows, steps, epochs = CFG["n_windows"], CFG["steps_per_window"], CFG["epochs"]
    C, lam = CFG["C"], CFG["lam"]

    grid = base_grid(H, W)
    gt = build_gt(H, W)
    poses = make_circular_poses(n_windows, radius=max(H, W) * 0.015)
    gt_img_ref = render(gt, grid).detach()

    model = GaussianScene2D.random(n_gauss, H, W, seed=seed)
    opt = torch.optim.Adam(model.params, lr=0.03)
    mode = "dense" if budget is None else "lif"
    gate = LIFGate(n_gauss, lam=lam, theta=THETA, mode=mode, budget=budget)

    opt_ms, fire_fracs, psnr_hist = [], [], []
    for ep in range(epochs):
        L_gt_prev, _ = gt_log_at_pose(gt, grid, poses[0])
        for k in range(1, n_windows):
            L_gt_curr, _ = gt_log_at_pose(gt, grid, poses[k])
            dE = accumulate_events(L_gt_prev, L_gt_curr, C).detach()

            # ---- gating pass (in a real rasterizer the stimulus falls out of the ----
            # ---- forward pass; here it is a separate no-grad pass, not timed as  ----
            # ---- part of the bounded optimization cost)                          ----
            with torch.no_grad():
                curr = render(model, grid + poses[k])
                prev = render(model, grid + poses[k - 1])
                dLhat0 = log_intensity(curr) - log_intensity(prev)
                w = gaussian_weights(model, grid + poses[k])
                s = gate.stimulus(w, (dLhat0 - dE).abs())
            mask = gate.step(s)
            fire_fracs.append(gate.last_fire_frac)

            # ---- bounded part: render/optimize ONLY the <=B active Gaussians ----
            t0 = time.perf_counter()
            cache_prev = render_inactive_cache(model, grid + poses[k - 1], mask)
            cache_curr = render_inactive_cache(model, grid + poses[k], mask)
            for _ in range(steps):
                opt.zero_grad()
                img_curr = render_active(model, grid + poses[k], mask, cache_curr)
                img_prev = render_active(model, grid + poses[k - 1], mask, cache_prev)
                dLhat = log_intensity(img_curr) - log_intensity(img_prev)
                loss = event_photometric_loss(dLhat, dE)
                if (k % 12) == 0:
                    loss = loss + 0.3 * anchor_loss(img_curr,
                                                    render(gt, grid + poses[k]).detach())
                loss.backward()
                gate.apply_gate(model.params, mask)
                frozen = LIFGate.freeze_snapshot(model.params)
                opt.step()
                LIFGate.restore_frozen(model.params, mask, frozen)
                model.clamp_(H, W)
            opt_ms.append((time.perf_counter() - t0) * 1e3)
            with torch.no_grad():
                psnr_hist.append(psnr(render(model, grid), gt_img_ref))
            L_gt_prev = L_gt_curr

    tail = psnr_hist[-10:]
    return dict(psnr=sum(tail) / len(tail),
                ms=sum(opt_ms) / len(opt_ms),
                active_pct=100 * sum(fire_fracs) / len(fire_fracs))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"C3 budgeted streaming | budgets={BUDGETS} | theta={THETA} | seeds={SEEDS}\n")

    rows, pts = [], []
    for B in BUDGETS:
        runs = [train_budget(B, seed) for seed in SEEDS]
        (pm, ps) = mean_std([r["psnr"] for r in runs])
        (mm, ms_) = mean_std([r["ms"] for r in runs])
        (am, as_) = mean_std([r["active_pct"] for r in runs])
        # A4: hardware-independent FLOP proxy of the bounded per-window optimisation
        mflops = budget_window_mflops(CFG["n_gauss"], am / 100.0 * CFG["n_gauss"],
                                      CFG["H"] * CFG["W"], CFG["steps_per_window"])
        label = "inf" if B is None else str(B)
        print(f"  B={label:>3}: {fmt(mm, ms_, 1):>13} ms/window | "
              f"PSNR {fmt(pm, ps)} dB | active {fmt(am, as_, 1)}% | {mflops:6.2f} MFLOP/win")
        rows.append([label, pm, ps, mm, ms_, am, as_, mflops])
        pts.append((label, pm, ps, mm, ms_))

    write_csv(os.path.join(OUTDIR, "budget.csv"),
              ["budget", "psnr_mean", "psnr_std", "ms_per_window_mean",
               "ms_per_window_std", "active_pct_mean", "active_pct_std", "mflops_per_window"], rows)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [p[0] for p in pts]
    x = range(len(pts))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.errorbar(x, [p[3] for p in pts], yerr=[p[4] for p in pts],
                fmt="o-", capsize=3, color="tab:green")
    a1.set_xticks(list(x), labels)
    a1.set_xlabel("budget B (max active Gaussians per window)")
    a1.set_ylabel("optimization ms / window")
    a1.set_title("C3 — per-window cost is bounded by B")
    a1.grid(alpha=0.3)
    a2.errorbar(x, [p[1] for p in pts], yerr=[p[2] for p in pts],
                fmt="s-", capsize=3, color="tab:blue")
    a2.set_xticks(list(x), labels)
    a2.set_xlabel("budget B")
    a2.set_ylabel("PSNR (dB)")
    a2.set_title("quality degrades gracefully as B shrinks")
    a2.grid(alpha=0.3)
    fig.suptitle(f"hard LIF budget (θ={THETA}, mean ± std over {len(SEEDS)} seeds)")
    fig.tight_layout()
    out = os.path.join(OUTDIR, "budget.png")
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
