"""
pipeline3d/train_real.py  --  streaming EvSpike-GS on REAL data (EventNeRF / DSEC).

*** STATUS: NOT RUN in this environment (no CUDA GPU, no dataset downloaded). ***
The code is complete and wired to the tested CPU modules (LIFGate, event loss, trainer),
but every number it would produce must come from an actual GPU run — none are claimed here.

Prerequisites on the GPU box:
    pip install gsplat h5py lpips
    # EventNeRF data:  https://github.com/r00tman/EventNeRF   -> --root/--scene
    # DSEC data:       https://dsec.ifi.uzh.ch                -> --root/--scene + poses.txt

Examples:
    python pipeline3d/train_real.py --dataset eventnerf --root data/eventnerf --scene lego \
        --theta 0.02 --budget 20000 --device cuda
    python pipeline3d/train_real.py --dataset dsec --root data/dsec --scene zurich_city_04_a \
        --intrinsics 556.0 556.0 336.0 240.0 --theta 0.02 --device cuda
"""
from __future__ import annotations
import os, sys, argparse
from typing import List, Optional

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import (Gaussians3D, render_torch, render_gsplat,   # noqa: E402
                      render_depth_gsplat)
from trainer3d import EvSpikeGSTrainer3D                          # noqa: E402
from datasets import (EventNeRFDataset, DSECDataset,              # noqa: E402
                      MVSECDataset, TUMVIEDataset)
from eval import (EfficiencyMeter, psnr, ssim, peak_vram_mb,      # noqa: E402
                  depth_rmse_vs_lidar)


def init_gaussians_from_points(points: torch.Tensor, n_max: int = 100_000,
                               scale: float = 0.02, brightness: float = 0.6) -> Gaussians3D:
    """
    Initialize Gaussians from a coarse point cloud (DSEC LiDAR/stereo, or EventNeRF
    pseudo-frame/COLMAP points). Subsamples to n_max; opacity starts low so the render
    is dim and events + anchors build the scene up (avoids early saturation).
    """
    if points.shape[0] > n_max:
        idx = torch.randperm(points.shape[0])[:n_max]
        points = points[idx]
    n = points.shape[0]
    log_scales = torch.log(torch.full((n, 3), float(scale)))
    quats = torch.zeros(n, 4); quats[:, 0] = 1.0
    opacity_logit = torch.full((n,), -2.5)
    b = torch.full((n,), float(brightness))
    return Gaussians3D(points.float(), log_scales, quats, opacity_logit, b)


def random_init_in_frustum(n: int, center: tuple, spread: float, seed: int = 0,
                           scale: float = 0.01) -> Gaussians3D:
    """Fallback init when no point cloud is available (EventNeRF object-centric scenes).
    EventNeRF-synthetic cameras orbit at radius ~0.9 with the object inside ~0.3 units of
    the origin, so spread ~0.5 and scale ~0.01 are the right magnitudes."""
    return Gaussians3D.random(n, center=center, spread=spread, scale=scale, seed=seed)


def estimate_background(anchor_img: torch.Tensor) -> float:
    """Median brightness of the image border — real captures rarely have black backgrounds,
    and compositing on the wrong background poisons the anchor loss."""
    b = torch.cat([anchor_img[0, :], anchor_img[-1, :],
                   anchor_img[:, 0], anchor_img[:, -1]])
    return float(b.median().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["eventnerf", "dsec", "mvsec", "tumvie"],
                    required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--intrinsics", type=float, nargs=4, default=None,
                    help="fx fy cx cy (DSEC: from cam_to_cam.yaml camRect0; TUM-VIE: "
                         "optional, defaults to the bundle's intrinsics.txt)")
    ap.add_argument("--points", type=str, default=None,
                    help="optional .npy (N,3) coarse point cloud for initialization")
    ap.add_argument("--n_gauss", type=int, default=50_000)
    ap.add_argument("--spread", type=float, default=0.5,
                    help="random-init cube side (EventNeRF object sits within ~0.3 of origin)")
    ap.add_argument("--max_scale", type=float, default=0.05,
                    help="world-space cap on Gaussian scale (anti-blur; ~object_radius/5)")
    ap.add_argument("--window_us", type=int, default=5000, help="DSEC window length")
    ap.add_argument("--n_windows", type=int, default=200,
                    help="EventNeRF/MVSEC/TUM-VIE: number of equal time windows")
    ap.add_argument("--t_start", type=float, default=None,
                    help="MVSEC/TUM-VIE: clip stream start (absolute seconds)")
    ap.add_argument("--t_stop", type=float, default=None,
                    help="MVSEC/TUM-VIE: clip stream stop (absolute seconds)")
    ap.add_argument("--C", type=float, default=0.25)
    ap.add_argument("--theta", type=float, default=1e-5,
                    help="LIF threshold; gsplat backend uses the gradient-norm stimulus, "
                         "whose scale is ~1e-5 (verified on EventNeRF chair)")
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--mode", type=str, default="lif",
                    choices=["lif", "dense", "binary", "topk"])
    ap.add_argument("--budget", type=int, default=None, help="hard active-set cap B")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=2, help="Adam steps per packet")
    ap.add_argument("--densify_every", type=int, default=500, help="packets between densify")
    ap.add_argument("--epochs", type=int, default=1, help="passes over the stream")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="outputs/real")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("train_real.py needs a CUDA GPU (gsplat backend). "
                         "This script was NOT run in the CPU-only development environment.")
    os.makedirs(args.outdir, exist_ok=True)

    # ---- dataset ---------------------------------------------------------------------
    if args.dataset == "eventnerf":
        ds = EventNeRFDataset(args.root, args.scene, n_windows=args.n_windows, C=args.C)
    elif args.dataset == "mvsec":
        ds = MVSECDataset(args.root, args.scene, n_windows=args.n_windows, C=args.C,
                          t_start_s=args.t_start, t_stop_s=args.t_stop)
    elif args.dataset == "tumvie":
        # bundle from tools/tumvie_convert.py; intrinsics fall back to intrinsics.txt.
        # No depth GT -> image-space evaluation only (see TUMVIEDataset docstring).
        ds = TUMVIEDataset(args.root, args.scene,
                           intrinsics=tuple(args.intrinsics) if args.intrinsics else None,
                           n_windows=args.n_windows, C=args.C,
                           t_start_s=args.t_start, t_stop_s=args.t_stop)
    else:
        if args.intrinsics is None:
            raise SystemExit("--intrinsics fx fy cx cy is required for DSEC")
        ds = DSECDataset(args.root, args.scene, tuple(args.intrinsics),
                         window_us=args.window_us, C=args.C)

    # ---- init ------------------------------------------------------------------------
    if args.points is not None:
        import numpy as np
        g = init_gaussians_from_points(torch.from_numpy(np.load(args.points)),
                                       n_max=args.n_gauss)
    elif args.dataset == "mvsec":
        # coarse point cloud from the first GT depth map (depth used for INIT only)
        pts = ds.init_points_from_depth(max_points=args.n_gauss)
        print(f"init from depth back-projection: {pts.shape[0]} points")
        g = init_gaussians_from_points(pts, n_max=args.n_gauss,
                                       scale=args.max_scale / 3)
    elif args.dataset == "dsec":
        # coarse point cloud along the WHOLE trajectory (driving: one frame's frustum
        # covers only the start; union of back-projections every ~Nth GT disparity map,
        # init only — exactly the stereo/LiDAR coarse-init the protocol prescribes)
        n_disp = len(ds.disparity_ts_raw())
        per = max(1, args.n_gauss // 16)
        pts = torch.cat([ds.init_points_from_disparity(k, max_points=per)
                         for k in range(0, n_disp, max(1, n_disp // 16))])
        print(f"init from {min(16, n_disp)} disparity frames: {pts.shape[0]} points")
        g = init_gaussians_from_points(pts, n_max=args.n_gauss,
                                       scale=args.max_scale / 3)
    else:
        g = random_init_in_frustum(args.n_gauss, center=(0, 0, 0), spread=args.spread,
                                   seed=args.seed)
    for p in g.params:
        p.data = p.data.to(args.device)

    bg = 0.0
    anchor_probe = None
    if getattr(ds, "anchor", None) is not None:
        anchor_probe = ds.anchor["image"]
    elif hasattr(ds, "anchor_at"):
        anchor_probe = ds.anchor_at(ds.t0)["image"]
    if anchor_probe is not None:
        bg = estimate_background(anchor_probe)
        print(f"estimated background brightness from anchor border: {bg:.3f}")
    trainer = EvSpikeGSTrainer3D(g, backend="torch", lr=args.lr, lam=args.lam,
                                 theta=args.theta, mode=args.mode, C=args.C,
                                 budget=args.budget, bg=bg, max_scale=args.max_scale,
                                 scene_extent=args.spread)
    pixel_mask = getattr(ds, "green_mask", None)
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(args.device)
        print("using green-Bayer pixel mask for the event loss")
    meter = EfficiencyMeter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- streaming loop --------------------------------------------------------------
    print(f"streaming {args.dataset}/{args.scene} | mode={args.mode} theta={args.theta} "
          f"budget={args.budget} | {g.n} Gaussians")
    packet_i = 0
    for ep in range(args.epochs):
        for packet in ds.stream():
            dE = packet.dE.to(args.device)
            kf = packet.keyframe
            if kf is not None:
                kf = {"cam": kf["cam"], "image": kf["image"].to(args.device)}
            fire, n_upd = trainer.streaming_step_events(
                packet.cam_prev, packet.cam_curr, dE,
                anchor=kf, steps=args.steps, pixel_mask=pixel_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            meter.update(fire, n_upd, trainer.last_step_ms)
            packet_i += 1
            if args.densify_every and packet_i % args.densify_every == 0:
                nc, ns, np_ = trainer.densify_and_prune(max_gaussians=2 * args.n_gauss)
                print(f"  packet {packet_i}: densify +{nc}c +{ns}s -{np_}p "
                      f"-> {trainer.g.n} Gaussians")
            if packet_i % 100 == 0:
                s = meter.summary()
                print(f"  packet {packet_i}: active {s['sparsity_pct']:.1f}% | "
                      f"{s['ms_per_packet']:.1f} ms/packet")

    # ---- save + report ----------------------------------------------------------------
    ckpt = os.path.join(args.outdir, f"{args.scene}_{args.mode}_theta{args.theta}.pt")
    torch.save({k: p.detach().cpu() for k, p in
                zip(("means", "log_scales", "quats", "opacity_logit", "brightness"),
                    trainer.g.params)}, ckpt)
    s = meter.summary()
    print(f"done | packets={packet_i} | active={s['sparsity_pct']:.1f}% | "
          f"{s['ms_per_packet']:.1f} ms/packet | peak VRAM={peak_vram_mb()} MB")
    print(f"checkpoint: {ckpt}")

    # DSEC: depth RMSE/L1 vs LiDAR-disparity GT (the geometry metric; no RGB anchor,
    # so image-space PSNR is not meaningful here — Prop. 2 scale ambiguity)
    if args.dataset == "dsec":
        import numpy as np
        n_disp = len(ds.disparity_ts_raw())
        stats = []
        for k in (n_disp // 4, n_disp // 2, 3 * n_disp // 4):
            dep_gt, cam_d, _ = ds.depth_from_disparity(k)
            dep_pred = render_depth_gsplat(trainer.g, cam_d).cpu()
            valid = torch.isfinite(dep_gt) & (dep_gt > 1.0) & (dep_gt < 80.0) \
                & (dep_pred > 0.5)
            m = depth_rmse_vs_lidar(dep_pred, torch.nan_to_num(dep_gt), valid)
            stats.append(m)
            print(f"depth vs LiDAR-GT @ frame {k}: RMSE {m['rmse']:.2f} m | "
                  f"L1 {m['l1']:.2f} m | {m['n_valid']} px")
        print(f"depth RMSE mean: {np.mean([s['rmse'] for s in stats]):.2f} m")

    # MVSEC: depth RMSE/L1 vs GT depth at several timestamps (the geometry metric)
    if args.dataset == "mvsec":
        import numpy as np
        stats = []
        for frac in (0.25, 0.5, 0.75):
            t_eval = ds.t0 + frac * (ds.t1 - ds.t0)
            dep_gt, cam_d, gap = ds.depth_at(t_eval)
            dep_pred = render_depth_gsplat(trainer.g, cam_d).cpu()
            valid = torch.isfinite(dep_gt) & (dep_gt > 0.1) & (dep_pred > 0.1)
            m = depth_rmse_vs_lidar(dep_pred, torch.nan_to_num(dep_gt), valid)
            stats.append(m)
            print(f"depth vs GT @ t+{frac:.2f}: RMSE {m['rmse']:.3f} m | "
                  f"L1 {m['l1']:.3f} m | {m['n_valid']} px | gt gap {gap*1e3:.0f} ms")
        print(f"depth RMSE mean: {np.mean([s['rmse'] for s in stats]):.3f} m")

    # rough quality check against the anchor RGB view (grayscale; scale fixed by anchor)
    if getattr(ds, "anchor", None) is not None:
        with torch.no_grad():
            img, _, _ = trainer.render(trainer.g, ds.anchor["cam"])
            ref = ds.anchor["image"].to(img.device)
            print(f"PSNR vs anchor RGB view (rough): {psnr(img, ref):.2f} dB | "
                  f"SSIM {ssim(img, ref):.3f}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
        a1.imshow(ref.cpu(), cmap="gray", vmin=0, vmax=1); a1.set_title("anchor RGB (gray)")
        a2.imshow(img.cpu(), cmap="gray", vmin=0, vmax=1); a2.set_title("EvSpike-GS render")
        for a in (a1, a2):
            a.axis("off")
        out = os.path.join(args.outdir, f"{args.scene}_anchor_view.png")
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"wrote {out}")
    print("full eval: held-out views with pipeline3d/eval.py (PSNR/SSIM/LPIPS), "
          "theta sweep with plot_pareto.py")


if __name__ == "__main__":
    main()
