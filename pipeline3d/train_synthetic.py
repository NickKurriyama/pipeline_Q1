"""
train_synthetic.py  --  A-Z runnable 3D smoke test (CPU, no gsplat, no dataset).

Builds a small ground-truth 3D scene (a few Gaussian blobs), orbits a pinhole camera around
it, generates EVENTS from the log-intensity change between consecutive views, and reconstructs
the scene from those events alone (+ one keyframe anchor) using the real 3D rasterizer and the
LIF-gated streaming trainer. This exercises the ENTIRE 3D pipeline logic end to end on a laptop.

For real data (EventNeRF/DSEC) and quality numbers, switch the backend to 'gsplat' on a GPU and
use pipeline3d/datasets.py (see README).

Run:  python pipeline3d/train_synthetic.py
"""
from __future__ import annotations
import os, sys, argparse, math
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import PinholeCamera, Gaussians3D, render_torch
from trainer3d import EvSpikeGSTrainer3D
from evspikegs.metrics import psnr

torch.manual_seed(0)


def build_gt():
    means = torch.tensor([[-0.25, 0.10, 0.0], [0.25, -0.05, 0.15],
                          [0.05, 0.28, -0.10], [-0.10, -0.25, 0.05]])
    log_scales = torch.log(torch.full((4, 3), 0.12))
    quats = torch.zeros(4, 4); quats[:, 0] = 1.0
    opacity = torch.full((4,), 2.0)
    brightness = torch.tensor([0.9, 0.8, 1.0, 0.75])
    return Gaussians3D(means, log_scales, quats, opacity, brightness, learnable=False)


def base_cam(H, W, radius=1.6, elev=0.15):
    return (radius, elev)


def jitter_cam(k, n, H, W, radius=1.6, elev=0.15, amp=0.14):
    """Small circular camera motion around a fixed viewpoint (event regime: small motion)."""
    ang = 2 * math.pi * k / n
    eye = (radius * math.sin(amp * math.cos(ang)),          # small horizontal parallax
           elev + amp * math.sin(ang),                       # small vertical parallax
           radius * math.cos(amp * math.cos(ang)))
    return PinholeCamera.look_at(eye, (0, 0, 0), (0, 1, 0), H, W, fov_deg=55.0)


def main():
    ap = argparse.ArgumentParser()
    # defaults = the configuration characterized by pipeline3d/plot_pareto.py:
    # theta=0.04 -> ~15-25% active-set at PSNR >= the dense baseline
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--n_gauss", type=int, default=120)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--mode", type=str, default="lif", choices=["lif", "dense", "binary", "topk"])
    ap.add_argument("--theta", type=float, default=0.04)
    ap.add_argument("--C", type=float, default=0.10)
    ap.add_argument("--budget", type=int, default=None,
                    help="hard cap B on active-set size per window (claim C3)")
    ap.add_argument("--densify", action="store_true",
                    help="clone/split/prune on the active set at the end of each epoch")
    ap.add_argument("--device", type=str, default="cpu",
                    help="'cpu' or 'cuda' (device-agnostic torch rasterizer)")
    ap.add_argument("--backend", type=str, default="torch", choices=["torch", "gsplat"],
                    help="'gsplat' needs CUDA + compiled gsplat (real EWA rasterizer)")
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    H, W = args.H, args.W
    gt = build_gt().to(args.device)
    cams = [jitter_cam(i, args.views, H, W) for i in range(args.views)]

    # ground-truth intensity images (also used for events + one anchor + eval)
    with torch.no_grad():
        gt_imgs = [render_torch(gt, c)[0] for c in cams]

    # init model from a coarse point cloud near the scene (emulates stereo/LiDAR init)
    model = Gaussians3D.random(args.n_gauss, center=(0, 0, 0), spread=0.9,
                               scale=0.10, seed=1).to(args.device)
    if args.backend == "gsplat" and args.device != "cuda":
        raise SystemExit("--backend gsplat requires --device cuda")
    trainer = EvSpikeGSTrainer3D(model, backend=args.backend, lr=0.02,
                                 theta=args.theta, mode=args.mode, C=args.C,
                                 budget=args.budget)
    anchor = {"cam": cams[0], "image": gt_imgs[0]}

    print(f"3D smoke test | {H}x{W}, {args.n_gauss} Gaussians, {args.views} views, mode={args.mode}")
    fire_hist, psnr_hist = [], []
    for ep in range(args.epochs):
        for k in range(1, args.views):
            fire, _ = trainer.streaming_step(
                cams[k - 1], cams[k], gt_imgs[k - 1], gt_imgs[k],
                anchor=(anchor if k == 1 else None), steps=args.steps)
            fire_hist.append(fire)
        if args.densify and ep < args.epochs - 1:
            nc, ns, np_ = trainer.densify_and_prune(max_gaussians=2 * args.n_gauss)
            print(f"  densify: +{nc} cloned, +{ns} split, -{np_} pruned "
                  f"-> {trainer.g.n} Gaussians")
        model = trainer.g   # densification may rebuild the Gaussian set
        with torch.no_grad():
            # evaluate on a held-out-ish view (mid orbit), using the training backend
            ev = args.views // 2
            recon = trainer.render(model, cams[ev])[0]
            p = psnr(recon, gt_imgs[ev])
            psnr_hist.append(p)
        print(f"  epoch {ep+1}/{args.epochs}: eval PSNR={p:5.2f} dB | "
              f"avg active-set={sum(fire_hist)/len(fire_hist)*100:5.1f}% | "
              f"last step {trainer.last_step_ms:.1f} ms")

    # save a GT-vs-recon panel at a few views
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        idxs = [0, args.views // 3, 2 * args.views // 3]
        fig, axes = plt.subplots(2, len(idxs), figsize=(3 * len(idxs), 6))
        with torch.no_grad():
            for j, vi in enumerate(idxs):
                axes[0, j].imshow(gt_imgs[vi].cpu(), cmap="gray", vmin=0, vmax=1)
                axes[0, j].set_title(f"GT view {vi}"); axes[0, j].axis("off")
                rec = trainer.render(model, cams[vi])[0].cpu()
                axes[1, j].imshow(rec, cmap="gray", vmin=0, vmax=1)
                axes[1, j].set_title(f"recon ({args.mode})"); axes[1, j].axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "recon3d.png"), dpi=110)
        print(f"  wrote {args.outdir}/recon3d.png")
    except Exception as e:
        print("  (figure skipped:", e, ")")

    print(f"\nFinal: PSNR={psnr_hist[-1]:.2f} dB, avg active-set="
          f"{sum(fire_hist)/len(fire_hist)*100:.1f}%  (mode={args.mode})")


if __name__ == "__main__":
    main()
