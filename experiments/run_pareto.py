"""
run_pareto.py  --  Claim C1: the quality-efficiency Pareto curve over the firing threshold.

Sweeps the LIF threshold theta, >=3 seeds per point, and plots reconstruction PSNR vs
active-set size, with the dense (theta -> 0) model as the reference. This is the single
figure the roadmap calls "the experiment that sells the paper" (section 5.4).

Run:  python experiments/run_pareto.py
Outputs: outputs/pareto.png, outputs/pareto.csv
"""
from __future__ import annotations
import os
from common import CFG, SEEDS, OUTDIR, mean_std, fmt, write_csv, smoothed_psnr

from demo import train
from evspikegs.metrics import ssim

THETAS = [0.02, 0.04, 0.08, 0.15, 0.25, 0.40]


def run_point(mode, theta):
    psnrs, fires, ssims = [], [], []
    for seed in SEEDS:
        cfg = dict(CFG); cfg["theta"] = theta
        r = train(mode, cfg["H"], cfg["W"], cfg["n_gauss"], cfg["n_windows"],
                  cfg["steps_per_window"], cfg["C"], cfg["lam"], cfg["theta"],
                  epochs=cfg["epochs"], seed=seed, verbose=False)
        psnrs.append(smoothed_psnr(r))
        fires.append(r["avg_fire"] * 100)
        ssims.append(ssim(r["recon"], r["gt"]))
    return mean_std(psnrs), mean_std(fires), mean_std(ssims)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"C1 Pareto sweep | thetas={THETAS} | seeds={SEEDS}\n")

    rows = []
    print("  dense reference (theta -> 0):")
    (dp, dps), (df, dfs), (ds, dss) = run_point("dense", 0.0)
    print(f"    PSNR {fmt(dp, dps)} dB @ {fmt(df, dfs, 1)}% active")
    rows.append(["dense", 0.0, dp, dps, df, dfs, ds, dss])

    pts = []
    for th in THETAS:
        (p, ps), (f, fs), (s, ss) = run_point("lif", th)
        print(f"  theta={th:<5}: PSNR {fmt(p, ps)} dB | SSIM {fmt(s, ss, 3)} | "
              f"active {fmt(f, fs, 1)}%")
        rows.append(["lif", th, p, ps, f, fs, s, ss])
        pts.append((th, p, ps, f, fs))

    write_csv(os.path.join(OUTDIR, "pareto.csv"),
              ["mode", "theta", "psnr_mean", "psnr_std",
               "active_pct_mean", "active_pct_std", "ssim_mean", "ssim_std"], rows)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = [f for _, _, _, f, _ in pts]
    ys = [p for _, p, _, _, _ in pts]
    xerr = [fs for *_, fs in pts]
    yerr = [ps for _, _, ps, _, _ in pts]
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="o-", capsize=3,
                color="tab:blue", label="EvSpike-GS (LIF, sweep θ)")
    for (th, p, _, f, _) in pts:
        ax.annotate(f"θ={th}", (f, p), textcoords="offset points",
                    xytext=(6, -4), fontsize=8)
    ax.axhline(dp, ls="--", color="tab:red", lw=1)
    ax.errorbar([df], [dp], xerr=[dfs], yerr=[dps], fmt="*", ms=14,
                color="tab:red", capsize=3, label="dense (θ→0)")
    ax.fill_between([0, 105], dp - 1.0, dp + 1.0, color="tab:red", alpha=0.08,
                    label="dense ± 1 dB")
    ax.set_xlim(0, 105)
    ax.set_xlabel("active-set size (% Gaussians updated per window)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"C1 — quality vs efficiency across firing threshold "
                 f"(mean ± std, {len(SEEDS)} seeds)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "pareto.png")
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
