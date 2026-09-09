"""
demo.py  --  EvSpike-GS mechanism demo (2D, CPU, runs A-Z)
==========================================================

What it does, end to end:
  1. Builds a fixed ground-truth 2D scene (a few blobs).
  2. Moves a virtual camera along a small circular trajectory and generates a stream of
     EVENTS from the log-intensity change between consecutive windows (event_data.py).
  3. Reconstructs the scene from those events with a set of 2D Gaussians, supervised ONLY by
     the differentiable event photometric loss (event_loss.py) + one keyframe anchor.
  4. Gates the optimization with the LIF mechanism (lif_gate.py): only "firing" Gaussians
     are updated each window.
  5. Repeats for several gate modes and reports the quality/efficiency trade-off, plus saves
     reconstructions and curves to ./outputs/.

This is a faithful *miniature* of the full method. It demonstrates the mechanism; it is NOT
the 3D reconstruction system (that is the research project -- see ./pipeline3d/README.md).

Run:  python demo.py
"""

from __future__ import annotations
import os
import argparse
import torch

from evspikegs.splat2d import GaussianScene2D, base_grid, render, log_intensity, gaussian_weights
from evspikegs.event_data import accumulate_events, make_circular_poses, gt_log_at_pose
from evspikegs.event_loss import event_photometric_loss, anchor_loss
from evspikegs.lif_gate import LIFGate
from evspikegs.metrics import psnr, mae

torch.manual_seed(0)


def build_gt(H, W):
    """A fixed ground-truth scene: a handful of bright blobs on a dark background."""
    mu = torch.tensor([
        [W * 0.30, H * 0.35], [W * 0.65, H * 0.30], [W * 0.50, H * 0.60],
        [W * 0.35, H * 0.70], [W * 0.70, H * 0.68],
    ])
    ls = torch.log(torch.full((mu.shape[0],), max(H, W) / 12.0))
    b = torch.tensor([0.9, 0.8, 1.0, 0.7, 0.85])
    o = torch.full((mu.shape[0],), 1.5)
    return GaussianScene2D(mu, ls, b, o, learnable=False)


def train(mode, H, W, n_gauss, n_windows, steps_per_window, C, lam, theta,
          epochs=4, anchor_every=12, anchor_w=0.3, verbose=True, seed=1,
          budget=None, topk_frac=0.30):
    grid = base_grid(H, W)
    gt = build_gt(H, W)
    # SMALL camera motion => only edges cross the contrast threshold => events are
    # spatially sparse, exactly the regime where gated updates matter.
    poses = make_circular_poses(n_windows, radius=max(H, W) * 0.015)

    gt_img_ref = render(gt, grid).detach()          # fixed-camera GT image for eval + anchor

    model = GaussianScene2D.random(n_gauss, H, W, seed=seed)
    opt = torch.optim.Adam(model.params, lr=0.03)
    gate = LIFGate(n_gauss, lam=lam, theta=theta, mode=mode,
                   budget=budget, topk_frac=topk_frac)

    fire_history, psnr_history = [], []
    total_updates = 0

    for ep in range(epochs):
        L_gt_prev, _ = gt_log_at_pose(gt, grid, poses[0])
        for k in range(1, n_windows):
            L_gt_curr, _ = gt_log_at_pose(gt, grid, poses[k])
            dE = accumulate_events(L_gt_prev, L_gt_curr, C).detach()

            for _ in range(steps_per_window):
                opt.zero_grad()
                img_curr = render(model, grid + poses[k])
                img_prev = render(model, grid + poses[k - 1])
                dLhat = log_intensity(img_curr) - log_intensity(img_prev)

                loss = event_photometric_loss(dLhat, dE)
                # periodic keyframe anchors (role of sparse RGB frames) fix radiance scale
                if (k % anchor_every) == 0:
                    loss = loss + anchor_w * anchor_loss(render(model, grid + poses[k]),
                                                         render(gt, grid + poses[k]).detach())
                loss.backward()

                # ---- LIF gating: choose the active set, then update only it ----
                with torch.no_grad():
                    w = gaussian_weights(model, grid + poses[k]).detach()
                    residual = (dLhat.detach() - dE).abs()
                    s = gate.stimulus(w, residual)
                mask = gate.step(s)
                gate.apply_gate(model.params, mask)
                frozen = LIFGate.freeze_snapshot(model.params)
                opt.step()
                LIFGate.restore_frozen(model.params, mask, frozen)
                model.clamp_(H, W)   # box constraints kill the saturation degenerate optimum
                total_updates += int(mask.sum().item())

            fire_history.append(gate.last_fire_frac)
            with torch.no_grad():
                psnr_history.append(psnr(render(model, grid), gt_img_ref))
            L_gt_prev = L_gt_curr

    with torch.no_grad():
        recon = render(model, grid).reshape(H, W)
    avg_fire = sum(fire_history) / max(1, len(fire_history))
    final_psnr = psnr_history[-1]
    if verbose:
        print(f"  [{mode:6s}] final PSNR = {final_psnr:5.2f} dB | "
              f"avg active-set = {avg_fire*100:5.1f}% | "
              f"total param-updates = {total_updates}")
    return dict(mode=mode, recon=recon, gt=gt_img_ref.reshape(H, W),
                psnr_hist=psnr_history, fire_hist=fire_history,
                final_psnr=final_psnr, avg_fire=avg_fire, total_updates=total_updates)


def save_figures(results, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("  (matplotlib unavailable, skipping figures:", e, ")")
        return

    # reconstructions
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    axes[0].imshow(results[0]["gt"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground truth"); axes[0].axis("off")
    for ax, r in zip(axes[1:], results):
        ax.imshow(r["recon"], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{r['mode']}\n{r['final_psnr']:.1f} dB, {r['avg_fire']*100:.0f}% active")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "reconstructions.png"), dpi=120)
    plt.close(fig)

    # quality vs efficiency curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        a1.plot(r["psnr_hist"], label=r["mode"])
        a2.plot([f * 100 for f in r["fire_hist"]], label=r["mode"])
    a1.set_title("Reconstruction PSNR vs window"); a1.set_xlabel("window"); a1.set_ylabel("PSNR (dB)"); a1.legend()
    a2.set_title("Active-set size (%) vs window"); a2.set_xlabel("window"); a2.set_ylabel("% Gaussians updated"); a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "curves.png"), dpi=120)
    plt.close(fig)
    print(f"  figures written to {outdir}/reconstructions.png and {outdir}/curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--W", type=int, default=48)
    ap.add_argument("--n_gauss", type=int, default=200)
    ap.add_argument("--windows", type=int, default=80)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--C", type=float, default=0.12, help="event contrast threshold")
    ap.add_argument("--lam", type=float, default=0.6, help="LIF leak")
    ap.add_argument("--theta", type=float, default=0.04, help="LIF firing threshold")
    ap.add_argument("--epochs", type=int, default=5, help="passes over the event stream")
    ap.add_argument("--modes", type=str, default="dense,lif,binary,topk")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=None,
                    help="hard cap B on active-set size per window (claim C3)")
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("EvSpike-GS 2D mechanism demo")
    print(f"  scene {args.H}x{args.W}, {args.n_gauss} Gaussians, {args.windows} event windows\n")
    print("Training under each gate mode (event-supervised, streaming):")

    results = []
    for mode in args.modes.split(","):
        r = train(mode.strip(), args.H, args.W, args.n_gauss, args.windows,
                  args.steps, args.C, args.lam, args.theta, epochs=args.epochs,
                  seed=args.seed, budget=args.budget)
        results.append(r)

    print("\nSummary (quality vs efficiency):")
    print(f"  {'mode':8s} {'PSNR(dB)':>9s} {'active-set%':>12s} {'updates':>10s}")
    for r in results:
        print(f"  {r['mode']:8s} {r['final_psnr']:9.2f} {r['avg_fire']*100:12.1f} {r['total_updates']:10d}")

    save_figures(results, args.outdir)
    print("\nTakeaway: LIF/binary/topk reach PSNR close to 'dense' while updating far fewer "
          "Gaussians per window -- the mechanism behind bounded-cost streaming reconstruction.")


if __name__ == "__main__":
    main()
