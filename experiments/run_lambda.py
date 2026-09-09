"""
run_lambda.py  --  A5 (claim C2b): isolate the LEAK from the threshold.

The C2 ablation (run_ablation.py) shows LIF > binary/top-k at matched sparsity, but a reviewer
can still ask: is it the *leaky temporal integration* that helps, or just the threshold? This
script answers by sweeping ONLY the leak lambda while holding the firing rate fixed.

Key fact (see lif_gate.py): 'lif' with lam=0 is V <- 0*V + s, i.e. fire iff s >= theta -- the
memoryless mask. So lambda=0 IS the "just a threshold" baseline, and lambda>0 adds temporal
memory. To separate leak from sparsity we calibrate theta PER lambda so every point sits at the
same active-set fraction f* (the paper default lam=0.6, theta=0.08), exactly the matched-sparsity
protocol used for C2.

Run:  python experiments/run_lambda.py
Outputs: outputs/lambda.csv, outputs/lambda.png (+ printed table)
"""
from __future__ import annotations
import os
from common import CFG, SEEDS, OUTDIR, mean_std, fmt, write_csv, smoothed_psnr

from demo import train
from evspikegs.metrics import ssim

LAMBDAS = [0.0, 0.3, 0.6, 0.9]
# probe grid is scaled per lambda by the steady-state gain 1/(1-lambda), because a leaky unit
# accumulates to ~s/(1-lambda); this keeps the grid bracketing f* for every lambda.
BASE_PROBE_THETAS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25]


def run_lam(lam, theta, seeds=SEEDS, epochs=None):
    cfg = dict(CFG)
    if epochs is not None:
        cfg["epochs"] = epochs
    runs = []
    for seed in seeds:
        r = train("lif", cfg["H"], cfg["W"], cfg["n_gauss"], cfg["n_windows"],
                  cfg["steps_per_window"], cfg["C"], lam, theta,
                  epochs=cfg["epochs"], seed=seed, verbose=False)
        r["ssim"] = ssim(r["recon"], r["gt"])
        runs.append(r)
    return runs


def agg(runs):
    p = mean_std([smoothed_psnr(r) for r in runs])
    s = mean_std([r["ssim"] for r in runs])
    f = mean_std([r["avg_fire"] * 100 for r in runs])
    return p, s, f


def calibrate_theta(lam, f_star_frac):
    """Pick theta (grid scaled by 1/(1-lam)) whose sparsity is closest to f*.
    Probes run at the FULL epoch count so the calibrated sparsity is measured in the same
    regime as the final runs -- this makes the sparsity genuinely matched across lambda
    (an epoch-1 probe fires more than the converged epoch-3 run, so it mis-calibrates)."""
    gain = 1.0 / max(1e-6, (1.0 - lam))
    best_th, best_gap, best_f = None, 1e9, None
    for base in BASE_PROBE_THETAS:
        th = base * gain
        probe = run_lam(lam, th, seeds=[SEEDS[0]])[0]   # epochs=None -> CFG epochs (full)
        gap = abs(probe["avg_fire"] - f_star_frac)
        if gap < best_gap:
            best_th, best_gap, best_f = th, gap, probe["avg_fire"]
    return best_th, best_f


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"A5 leak-lambda sweep (theta per-lambda toward f*) | lambdas={LAMBDAS} | seeds={SEEDS}\n")

    # reference active-set fraction f* = the paper default (lam=0.6, theta=0.08)
    ref_runs = run_lam(0.6, CFG["theta"])
    f_star = sum(r["avg_fire"] for r in ref_runs) / len(ref_runs)
    print(f"  target sparsity f* = {f_star*100:.1f}%  (from lam=0.6, theta={CFG['theta']})\n")

    header = f"  {'lambda':>6} {'theta':>7} {'PSNR (dB)':>16} {'SSIM':>16} {'active %':>14} {'note':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows, res = [], {}
    for lam in LAMBDAS:
        if abs(lam - 0.6) < 1e-9:
            th, runs = CFG["theta"], ref_runs
        else:
            th, _ = calibrate_theta(lam, f_star)
            runs = run_lam(lam, th)
        (pm, ps), (sm, ss), (fm, fs) = agg(runs)
        note = "memoryless" if lam == 0.0 else ("ours" if abs(lam - 0.6) < 1e-9 else "")
        print(f"  {lam:>6.2f} {th:>7.3f} {fmt(pm, ps):>16} {fmt(sm, ss, 3):>16} "
              f"{fmt(fm, fs, 1):>14} {note:>13}")
        rows.append([lam, th, pm, ps, sm, ss, fm, fs])
        res[lam] = (pm, ps, fm, fs)

    write_csv(os.path.join(OUTDIR, "lambda.csv"),
              ["lambda", "theta", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
               "active_pct_mean", "active_pct_std"], rows)

    p0, p6 = res[0.0][0], res[0.6][0]
    print(f"\n  A5 margin: lam=0.6 - lam=0 (memoryless) = {p6 - p0:+.2f} dB "
          f"(target f*~{f_star*100:.0f}%; see active% per row)")
    print("  NOTE (honest): PSNR here tracks active% almost 1:1, and theta-calibration cannot")
    print("  hold sparsity fixed across lambda in this toy, so the leak's INDEPENDENT effect is")
    print("  NOT isolated -- this sweep is inconclusive for the leak's quality contribution.")
    print("  The robust matched-sparsity evidence is real-data LIF vs fixed-k top-k (tab:abl).")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ys = [res[l][0] for l in LAMBDAS]
        es = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.errorbar(LAMBDAS, ys, yerr=es, fmt="o-", capsize=3, color="tab:blue")
        ax.annotate("memoryless\nmask", (0.0, ys[0]), textcoords="offset points",
                    xytext=(8, -4), fontsize=8, color="tab:red")
        ax.set_xlabel("leak  λ   (0 = memoryless mask)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"A5 — leak sweep (theta per-lambda toward f*~{f_star*100:.0f}%, "
                     f"{len(SEEDS)} seeds)")
        ax.grid(alpha=0.3); fig.tight_layout()
        out = os.path.join(OUTDIR, "lambda.png")
        fig.savefig(out, dpi=130)
        print(f"  wrote {os.path.relpath(out, os.path.dirname(OUTDIR))}")
    except Exception as e:
        print("  (plot skipped:", e, ")")


if __name__ == "__main__":
    main()
