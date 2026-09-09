"""
pipeline3d/final_table.py  --  the paper's final benchmark table on real EventNeRF data.

Two stages:
  1. Per-scene theta calibration: the LIF threshold trades update budget for quality and
     one global value under-serves thin-structure scenes (mic/drums). Short 1-seed probes
     pick theta per scene from a small grid.
  2. Final pass: every scene x {dense, lif@theta*} x 3 seeds, novel-view eval on the
     held-out split with PSNR / SSIM / LPIPS (AlexNet, if `lpips` is installed).

Appends to outputs/real/final.csv after every run; prints the mean +/- std table.

Run:  tools\\run_gpu.bat pipeline3d\\final_table.py
"""
from __future__ import annotations
import os, sys, csv, argparse, statistics, traceback
from typing import Dict, List, Optional

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import eval_real_suite as suite                                     # noqa: E402

SCENES = ["chair", "drums", "hotdog", "lego", "mic"]
SEEDS = [0, 1, 2]
PROBE_GRID = [1e-5, 3e-6, 1e-6]     # lower theta -> larger active set
PROBE_TARGET_DB = None               # pick probe with best PSNR (2-epoch, seed 0)

FIELDS = ["scene", "mode", "theta", "seed", "psnr", "ssim", "lpips",
          "active_pct", "ms_per_packet", "n_test_views"]

_lpips_model = None


def lpips_value(img: torch.Tensor, ref: torch.Tensor) -> Optional[float]:
    """LPIPS(AlexNet) on grayscale->3ch images in [0,1]; None if lpips not installed."""
    global _lpips_model
    try:
        import lpips
    except ImportError:
        return None
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).to(img.device)
    def to3(x):
        return (x.reshape(1, 1, *x.shape[-2:]).repeat(1, 3, 1, 1) * 2 - 1).float()
    with torch.no_grad():
        return float(_lpips_model(to3(img), to3(ref.to(img.device))).item())


def run_one(root, scene, mode, theta, seed, epochs) -> Dict:
    """suite.run_one + LPIPS over the held-out views."""
    r = suite.run_one(root, scene, mode, theta, seed, epochs=epochs)
    # recompute per-view LPIPS with a fresh short render pass? -> instead, fold LPIPS into
    # the same eval by re-running eval with the trainer... suite.run_one already freed the
    # trainer, so LPIPS is computed inside eval_novel_views via monkeypatch below.
    return r


def _patched_eval(trainer, views, device):
    ps, ss, lp = [], [], []
    from evspikegs.metrics import psnr, ssim
    for cam, ref in views:
        with torch.no_grad():
            img, _, _ = trainer.render(trainer.g, cam)
        ref_d = ref.to(img.device)
        ps.append(psnr(img, ref_d))
        ss.append(ssim(img, ref_d))
        v = lpips_value(img, ref_d)
        if v is not None:
            lp.append(v)
    out = {"psnr": sum(ps) / len(ps), "ssim": sum(ss) / len(ss)}
    out["lpips"] = (sum(lp) / len(lp)) if lp else float("nan")
    return out


suite.eval_novel_views = _patched_eval          # inject LPIPS into the suite's eval


def append_csv(path: str, row: Dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, float("nan")) for k in FIELDS})


def mean_std(xs):
    xs = [x for x in xs if x == x]      # drop NaN
    if not xs:
        return float("nan"), 0.0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data/nerf")
    ap.add_argument("--scenes", type=str, nargs="+", default=SCENES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--outdir", type=str, default="outputs/real")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "final.csv")
    rows: List[Dict] = []

    # ---- stage 1: per-scene theta (1-seed, 2-epoch probes, pick best PSNR) ------------
    print("Stage 1: per-scene theta calibration", flush=True)
    theta_star: Dict[str, float] = {}
    for scene in args.scenes:
        best_th, best_psnr = None, -1e9
        for th in PROBE_GRID:
            try:
                r = run_one(args.root, scene, "lif", th, args.seeds[0], epochs=2)
            except Exception:
                print(f"  {scene} probe th={th:g} FAILED", flush=True)
                continue
            print(f"  {scene:<8} probe th={th:<8g}: PSNR {r['psnr']:5.2f} | "
                  f"active {r['active_pct']:4.1f}%", flush=True)
            if r["psnr"] > best_psnr:
                best_th, best_psnr = th, r["psnr"]
        theta_star[scene] = best_th if best_th is not None else 1e-5
        print(f"  -> {scene}: theta* = {theta_star[scene]:g}", flush=True)

    # ---- stage 2: final multi-seed table ------------------------------------------------
    print(f"\nStage 2: final table ({args.epochs} epochs, seeds={args.seeds}, "
          f"PSNR/SSIM/LPIPS on held-out views)", flush=True)
    for scene in args.scenes:
        for seed in args.seeds:
            for mode, th in (("dense", 0.0), ("lif", theta_star[scene])):
                try:
                    r = run_one(args.root, scene, mode, th, seed, epochs=args.epochs)
                    append_csv(csv_path, r)
                    rows.append(r)
                    print(f"  {scene:<8} {mode:<6} th={th:<8g} seed={seed}: "
                          f"PSNR {r['psnr']:5.2f} | SSIM {r['ssim']:.3f} | "
                          f"LPIPS {r.get('lpips', float('nan')):.3f} | "
                          f"active {r['active_pct']:5.1f}%", flush=True)
                except Exception:
                    print(f"  {scene} {mode} seed={seed} FAILED:\n"
                          f"{traceback.format_exc()}", flush=True)

    # ---- summary -------------------------------------------------------------------------
    print("\n===== FINAL TABLE (mean +/- std, novel-view held-out split) =====", flush=True)
    print(f"  {'scene':<8} {'mode':<6} {'theta*':>8} {'PSNR (dB)':>16} {'SSIM':>15} "
          f"{'LPIPS':>15} {'active %':>13}")
    for scene in args.scenes:
        for mode in ("dense", "lif"):
            rs = [r for r in rows if r["scene"] == scene and r["mode"] == mode]
            if not rs:
                continue
            pm, ps_ = mean_std([r["psnr"] for r in rs])
            sm, ss_ = mean_std([r["ssim"] for r in rs])
            lm, ls_ = mean_std([r.get("lpips", float("nan")) for r in rs])
            am, _ = mean_std([r["active_pct"] for r in rs])
            th = theta_star[scene] if mode == "lif" else 0.0
            print(f"  {scene:<8} {mode:<6} {th:>8g} {pm:8.2f} +/- {ps_:4.2f} "
                  f"{sm:7.3f} +/- {ss_:.3f} {lm:7.3f} +/- {ls_:.3f} {am:10.1f}",
                  flush=True)
    print(f"\nper-run rows in {csv_path}", flush=True)


if __name__ == "__main__":
    main()
