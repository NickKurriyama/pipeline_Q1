"""
run_ablation.py  --  Claim C2: leaky temporal integration beats memoryless gating
                     AT MATCHED SPARSITY.

Protocol (roadmap section 5.4):
  1. Run LIF (ours) over >=3 seeds -> measure its average active-set fraction f*.
  2. Calibrate the ablations to the SAME sparsity: binary mask (threshold chosen by a
     1-seed probe grid so its sparsity is closest to f*), and top-k with k = f* * N.
  3. Run every mode over the same seeds; report PSNR / SSIM / sparsity as mean ± std.

Run:  python experiments/run_ablation.py
Outputs: outputs/ablation.csv (+ printed table)
"""
from __future__ import annotations
import os
from common import CFG, SEEDS, OUTDIR, mean_std, fmt, write_csv, smoothed_psnr

from demo import train
from evspikegs.metrics import ssim

BINARY_PROBE_THETAS = [0.01, 0.02, 0.04, 0.08, 0.15, 0.30]


def run_mode(mode, theta=None, topk_frac=0.30, seeds=SEEDS, epochs=None):
    cfg = dict(CFG)
    if theta is not None:
        cfg["theta"] = theta
    if epochs is not None:
        cfg["epochs"] = epochs
    out = []
    for seed in seeds:
        r = train(mode, cfg["H"], cfg["W"], cfg["n_gauss"], cfg["n_windows"],
                  cfg["steps_per_window"], cfg["C"], cfg["lam"], cfg["theta"],
                  epochs=cfg["epochs"], seed=seed, verbose=False, topk_frac=topk_frac)
        r["ssim"] = ssim(r["recon"], r["gt"])
        out.append(r)
    return out


def agg(runs):
    p = mean_std([smoothed_psnr(r) for r in runs])
    s = mean_std([r["ssim"] for r in runs])
    f = mean_std([r["avg_fire"] * 100 for r in runs])
    u = mean_std([float(r["total_updates"]) for r in runs])
    return p, s, f, u


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"C2 ablation at matched sparsity | seeds={SEEDS}\n")

    print("  [1/3] LIF (ours) ...")
    lif_runs = run_mode("lif")
    f_star = sum(r["avg_fire"] for r in lif_runs) / len(lif_runs)
    print(f"        avg active-set f* = {f_star*100:.1f}%")

    print("  [2/3] calibrating binary mask to f* (1-seed probes) ...")
    best_th, best_gap = None, 1e9
    for th in BINARY_PROBE_THETAS:
        probe = run_mode("binary", theta=th, seeds=[SEEDS[0]], epochs=1)[0]
        gap = abs(probe["avg_fire"] - f_star)
        if gap < best_gap:
            best_th, best_gap = th, gap
    print(f"        binary theta = {best_th} (closest probe sparsity)")

    print("  [3/3] full runs: dense / binary@matched / topk@matched ...")
    results = {
        "dense (theta->0)": run_mode("dense"),
        "lif (ours)": lif_runs,
        "binary @ matched": run_mode("binary", theta=best_th),
        "topk @ matched": run_mode("topk", topk_frac=f_star),
    }

    header = f"  {'mode':<18} {'PSNR (dB)':>16} {'SSIM':>16} {'active %':>14} {'updates':>20}"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for name, runs in results.items():
        (pm, ps), (sm, ss), (fm, fs), (um, us) = agg(runs)
        print(f"  {name:<18} {fmt(pm, ps):>16} {fmt(sm, ss, 3):>16} "
              f"{fmt(fm, fs, 1):>14} {fmt(um, us, 0):>20}")
        rows.append([name, pm, ps, sm, ss, fm, fs, um, us])

    write_csv(os.path.join(OUTDIR, "ablation.csv"),
              ["mode", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
               "active_pct_mean", "active_pct_std", "updates_mean", "updates_std"], rows)

    lif_p = agg(lif_runs)[0][0]
    bin_p = agg(results["binary @ matched"])[0][0]
    print(f"\n  C2 margin: LIF - binary = {lif_p - bin_p:+.2f} dB at ~equal sparsity "
          f"(leaky temporal integration is what the memoryless mask is missing).")


if __name__ == "__main__":
    main()
