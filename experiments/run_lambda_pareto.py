"""
run_lambda_pareto.py -- A5b: does the LEAK actually help QUALITY? (rigorous overlay)

The single-point sweep (run_lambda.py) was inconclusive because theta could not hold active%
fixed across lambda, so PSNR just tracked sparsity. This instead traces a FULL PSNR-vs-active%
Pareto curve for each lambda (sweep theta per lambda), then compares the curves AT MATCHED
active% by interpolation:

  - if the leaky curves (lambda>0) sit ABOVE the memoryless curve (lambda=0) at equal active%,
    the leak genuinely improves quality;
  - if they overlap, quality is set by sparsity alone and the leak's value is elsewhere
    (bounded-backlog / budget, Prop.4) -- not a quality boost.

Run:  python experiments/run_lambda_pareto.py
Outputs: outputs/lambda_pareto.csv, outputs/lambda_pareto.png (+ printed verdict)
"""
from __future__ import annotations
import os
import numpy as np
from common import CFG, SEEDS, OUTDIR, mean_std, fmt, write_csv, smoothed_psnr

from demo import train

LAMBDAS = [0.0, 0.3, 0.6, 0.9]
BASE_THETAS = [0.02, 0.04, 0.08, 0.16, 0.30]     # scaled per-lambda by 1/(1-lambda)


def run_point(lam, theta):
    ps, fs = [], []
    for seed in SEEDS:
        r = train("lif", CFG["H"], CFG["W"], CFG["n_gauss"], CFG["n_windows"],
                  CFG["steps_per_window"], CFG["C"], lam, theta,
                  epochs=CFG["epochs"], seed=seed, verbose=False)
        ps.append(smoothed_psnr(r))
        fs.append(r["avg_fire"] * 100)
    return mean_std(ps), mean_std(fs)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"A5b leak Pareto-overlay | lambdas={LAMBDAS} | thetas x1/(1-lam) | seeds={SEEDS}\n")

    curves, rows = {}, []
    for lam in LAMBDAS:
        gain = 1.0 / max(1e-6, 1.0 - lam)
        pts = []
        for base in BASE_THETAS:
            th = base * gain
            (pm, psd), (fm, fsd) = run_point(lam, th)
            pts.append((fm, pm))
            rows.append([lam, th, pm, psd, fm, fsd])
            print(f"  lam={lam:<4} th={th:6.3f} -> active {fm:5.1f}%  PSNR {pm:5.2f} +/- {psd:.2f}",
                  flush=True)
        curves[lam] = sorted(pts)   # sort by active%
        print(flush=True)

    write_csv(os.path.join(OUTDIR, "lambda_pareto.csv"),
              ["lambda", "theta", "psnr_mean", "psnr_std",
               "active_pct_mean", "active_pct_std"], rows)

    # --- quantitative verdict: interpolate each lambda>0 curve onto the memoryless grid ---
    base = curves[0.0]
    bx = np.array([p[0] for p in base]); by = np.array([p[1] for p in base])
    print("  VERDICT (PSNR at matched active%, LIF minus memoryless lambda=0):")
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        cx = np.array([p[0] for p in curves[lam]]); cy = np.array([p[1] for p in curves[lam]])
        lo, hi = max(bx.min(), cx.min()), min(bx.max(), cx.max())
        if hi <= lo:
            print(f"    lam={lam}: no active% overlap with memoryless -- cannot compare")
            continue
        grid = np.linspace(lo, hi, 25)
        d = np.interp(grid, cx, cy) - np.interp(grid, bx, by)
        tag = "leak HELPS" if d.mean() > 0.3 else ("leak HURTS" if d.mean() < -0.3
                                                   else "NO effect (overlap)")
        print(f"    lam={lam}: mean Delta = {d.mean():+.2f} dB (min {d.min():+.2f}, "
              f"max {d.max():+.2f}) over active% in [{lo:.0f},{hi:.0f}]  => {tag}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for lam in LAMBDAS:
            xs = [p[0] for p in curves[lam]]; ys = [p[1] for p in curves[lam]]
            lbl = f"lambda={lam}" + (" (memoryless)" if lam == 0.0 else
                                     (" (ours)" if lam == 0.6 else ""))
            ax.plot(xs, ys, "o-", label=lbl)
        ax.set_xlabel("active-set size (% Gaussians updated per window)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"A5b -- leak Pareto overlay (theta swept per lambda, {len(SEEDS)} seeds)")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        out = os.path.join(OUTDIR, "lambda_pareto.png")
        fig.savefig(out, dpi=130)
        print(f"\n  wrote {out}")
    except Exception as e:
        print("  (plot skipped:", e, ")")


if __name__ == "__main__":
    main()
