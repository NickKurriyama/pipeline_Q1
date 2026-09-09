"""
pipeline3d/eval_real_suite.py  --  the paper-grade experiment suite on real EventNeRF data.

What it does (GPU, gsplat backend):
  Phase 1: for every downloaded scene x {dense, lif} x seeds -> train the streaming
           system, then evaluate NOVEL-VIEW PSNR/SSIM on the held-out `test/` split
           (renders at the test poses, grayscale, against the test RGB images).
  Phase 2: C2 ablation on one scene: binary mask (threshold calibrated by probes to
           match LIF's sparsity) and top-k (k = LIF's average firing fraction),
           same seeds — leaky integration vs memoryless gating on REAL data.

Results are appended to outputs/real/suite.csv after every run (crash-safe) and a
mean +/- std summary table is printed at the end.

Run:  tools\\run_gpu.bat pipeline3d\\eval_real_suite.py
"""
from __future__ import annotations
import os, sys, csv, glob, argparse, statistics, traceback
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import PinholeCamera, Gaussians3D                    # noqa: E402
from trainer3d import EvSpikeGSTrainer3D                           # noqa: E402
from datasets import EventNeRFDataset                              # noqa: E402
from train_real import estimate_background                         # noqa: E402
from evspikegs.metrics import psnr, ssim                           # noqa: E402

SCENES = ["chair", "drums", "hotdog", "lego", "mic"]
SEEDS = [0, 1, 2]
LIF_THETA = 1e-5          # gradient-norm stimulus scale (calibrated on chair)
N_GAUSS = 20_000
N_WINDOWS = 300
EPOCHS = 6
SPREAD, MAX_SCALE = 0.5, 0.05
DENSIFY_EVERY = 500


# ------------------------------------------------------------------------------------------
# held-out test views (novel-view evaluation — the proper protocol)
# ------------------------------------------------------------------------------------------
def load_test_views(root: str, scene: str, ds: EventNeRFDataset):
    """Load the held-out split: (PinholeCamera, grayscale image) per view.
    Uses `test/`; falls back to `validation/` (some scenes, e.g. drums, ship no test)."""
    import matplotlib.image as mpimg
    tdir = os.path.join(root, scene, "test")
    if not os.path.isdir(os.path.join(tdir, "pose")):
        tdir = os.path.join(root, scene, "validation")
    views = []
    for pf in sorted(glob.glob(os.path.join(tdir, "pose", "r_*.txt"))):
        idx = os.path.basename(pf)[2:-4]
        rf = os.path.join(tdir, "rgb", f"r_{idx}.png")
        if not os.path.exists(rf):
            continue
        m = np.loadtxt(pf).reshape(4, 4)
        R_w2c = m[:3, :3].T
        t_w2c = -R_w2c @ m[:3, 3]
        cam = PinholeCamera(ds.H, ds.W, ds.fx, ds.fy, ds.cx, ds.cy,
                            torch.tensor(R_w2c, dtype=torch.float32),
                            torch.tensor(t_w2c, dtype=torch.float32))
        img = mpimg.imread(rf)
        if img.ndim == 3:
            img = img[..., :3].mean(-1)
        views.append((cam, torch.from_numpy(np.ascontiguousarray(img)).float()))
    return views


@torch.no_grad()
def eval_novel_views(trainer, views, device) -> Dict[str, float]:
    ps, ss = [], []
    for cam, ref in views:
        img, _, _ = trainer.render(trainer.g, cam)
        ref = ref.to(img.device)
        ps.append(psnr(img, ref))
        ss.append(ssim(img, ref))
    return {"psnr": sum(ps) / len(ps), "ssim": sum(ss) / len(ss)}


# ------------------------------------------------------------------------------------------
# one training run
# ------------------------------------------------------------------------------------------
def run_one(root: str, scene: str, mode: str, theta: float, seed: int,
            epochs: int = EPOCHS, topk_frac: float = 0.3,
            device: str = "cuda") -> Dict[str, float]:
    ds = EventNeRFDataset(root, scene, n_windows=N_WINDOWS)
    torch.manual_seed(seed)
    g = Gaussians3D.random(N_GAUSS, center=(0, 0, 0), spread=SPREAD,
                           scale=0.01, seed=seed).to(device)
    bg = estimate_background(ds.anchor["image"]) if ds.anchor is not None else 0.0
    trainer = EvSpikeGSTrainer3D(g, backend="gsplat", theta=theta, mode=mode,
                                 C=ds.C, bg=bg, max_scale=MAX_SCALE,
                                 scene_extent=SPREAD)
    trainer.gate.topk_frac = topk_frac
    mask = ds.green_mask.to(device)

    fire_hist, ms_hist = [], []
    packet_i = 0
    for _ in range(epochs):
        for packet in ds.stream():
            kf = packet.keyframe
            if kf is not None:
                kf = {"cam": kf["cam"], "image": kf["image"].to(device)}
            f, _n = trainer.streaming_step_events(
                packet.cam_prev, packet.cam_curr, packet.dE.to(device),
                anchor=kf, steps=2, pixel_mask=mask)
            fire_hist.append(f)
            ms_hist.append(trainer.last_step_ms)
            packet_i += 1
            if packet_i % DENSIFY_EVERY == 0:
                trainer.densify_and_prune(max_gaussians=2 * N_GAUSS)

    views = load_test_views(root, scene, ds)
    m = eval_novel_views(trainer, views, device)
    m.update(scene=scene, mode=mode, theta=theta, seed=seed,
             active_pct=100 * sum(fire_hist) / len(fire_hist),
             ms_per_packet=sum(ms_hist) / len(ms_hist),
             n_test_views=len(views))
    del trainer, g
    torch.cuda.empty_cache()
    return m


# ------------------------------------------------------------------------------------------
# suite driver
# ------------------------------------------------------------------------------------------
FIELDS = ["scene", "mode", "theta", "seed", "psnr", "ssim",
          "active_pct", "ms_per_packet", "n_test_views"]


def append_csv(path: str, row: Dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


def mean_std(xs: List[float]):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data/nerf")
    ap.add_argument("--scenes", type=str, nargs="+", default=SCENES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--ablation_scene", type=str, default="chair")
    ap.add_argument("--skip_phase1", action="store_true")
    ap.add_argument("--skip_phase2", action="store_true")
    ap.add_argument("--outdir", type=str, default="outputs/real")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "suite.csv")
    rows: List[Dict] = []

    def do(scene, mode, theta, seed, topk_frac=0.3):
        try:
            r = run_one(args.root, scene, mode, theta, seed,
                        epochs=args.epochs, topk_frac=topk_frac)
            append_csv(csv_path, r)
            rows.append(r)
            print(f"  {scene:<8} {mode:<7} th={theta:<8g} seed={seed}: "
                  f"PSNR {r['psnr']:5.2f} | SSIM {r['ssim']:.3f} | "
                  f"active {r['active_pct']:5.1f}% | {r['ms_per_packet']:.0f} ms/pkt",
                  flush=True)
            return r
        except Exception:
            print(f"  {scene} {mode} seed={seed} FAILED:\n{traceback.format_exc()}",
                  flush=True)
            return None

    # ---- Phase 1: all scenes, dense vs LIF, multi-seed --------------------------------
    if not args.skip_phase1:
        print(f"Phase 1: {args.scenes} x [dense, lif] x seeds={args.seeds} "
              f"({args.epochs} epochs, {N_WINDOWS} windows, novel-view eval on test/)",
              flush=True)
        for scene in args.scenes:
            for seed in args.seeds:
                do(scene, "dense", 0.0, seed)
                do(scene, "lif", LIF_THETA, seed)

    # ---- Phase 2: C2 ablation at matched sparsity on one scene ------------------------
    if args.skip_phase2:
        _summary(rows, csv_path)
        return
    print(f"\nPhase 2: C2 ablation on '{args.ablation_scene}' (binary/topk @ matched "
          f"sparsity)", flush=True)
    lif_rows = [r for r in rows if r and r["scene"] == args.ablation_scene
                and r["mode"] == "lif"]
    if lif_rows:
        f_star = statistics.mean(r["active_pct"] for r in lif_rows) / 100.0
    else:
        f_star = 0.054      # measured on chair earlier
    print(f"  target sparsity f* = {f_star*100:.1f}%", flush=True)

    print("  calibrating binary threshold (1-seed, 2-epoch probes)...", flush=True)
    best_th, best_gap = None, 1e9
    for th in [3e-6, 1e-5, 3e-5, 1e-4]:
        r = None
        try:
            r = run_one(args.root, args.ablation_scene, "binary", th,
                        args.seeds[0], epochs=2)
        except Exception:
            continue
        gap = abs(r["active_pct"] / 100.0 - f_star)
        print(f"    probe th={th:g}: active {r['active_pct']:.1f}%", flush=True)
        if gap < best_gap:
            best_th, best_gap = th, gap
    print(f"  binary theta = {best_th:g}", flush=True)

    for seed in args.seeds:
        do(args.ablation_scene, "binary", best_th, seed)
        do(args.ablation_scene, "topk", 0.0, seed, topk_frac=f_star)

    _summary(rows, csv_path)


def _summary(rows: List[Dict], csv_path: str) -> None:
    print("\n===== SUMMARY (mean +/- std over seeds, novel-view held-out split) =====",
          flush=True)
    keys = sorted({(r["scene"], r["mode"]) for r in rows if r})
    print(f"  {'scene':<8} {'mode':<7} {'PSNR (dB)':>16} {'SSIM':>14} {'active %':>14}")
    for scene, mode in keys:
        rs = [r for r in rows if r and r["scene"] == scene and r["mode"] == mode]
        pm, ps_ = mean_std([r["psnr"] for r in rs])
        sm, ss_ = mean_std([r["ssim"] for r in rs])
        am, as_ = mean_std([r["active_pct"] for r in rs])
        print(f"  {scene:<8} {mode:<7} {pm:8.2f} +/- {ps_:4.2f} "
              f"{sm:8.3f} +/- {ss_:.3f} {am:8.1f} +/- {as_:4.1f}")
    print(f"\nper-run rows in {csv_path}", flush=True)


if __name__ == "__main__":
    main()
