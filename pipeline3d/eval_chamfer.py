"""
pipeline3d/eval_chamfer.py -- geometry evaluation on EventNeRF-synthetic: Chamfer/P2P
between the reconstructed Gaussian cloud and the ground-truth mesh vertices exported
from the original .blend scenes (tools/blend_export_points.py).

Frame note (stated, not hidden): EventNeRF normalises its cameras, so the reconstruction
lives in a similarity-transformed copy of the Blender world. Both clouds are therefore
brought to a canonical frame (centroid at origin, unit RMS radius) and we report
SCALE-NORMALISED Chamfer + per-direction point-to-point means — valid for comparing
methods on the same scene, which is its role in the protocol.

Run:  tools\\run_gpu.bat pipeline3d\\eval_chamfer.py           # trains lif+dense per scene
      tools\\run_gpu.bat pipeline3d\\eval_chamfer.py --scenes chair lego
"""
from __future__ import annotations
import os, sys, csv, argparse
from typing import Dict, List

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import eval_real_suite as suite                                     # noqa: E402
from eval import chamfer_distance                                   # noqa: E402

THETA = {"chair": 0.1, "hotdog": 0.1, "lego": 0.1, "drums": 0.1, "mic": 0.1}


def normalise(P: torch.Tensor) -> torch.Tensor:
    """Canonical frame: centroid at origin, unit RMS radius."""
    P = P - P.mean(dim=0, keepdim=True)
    return P / P.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-9)


def gaussians_to_cloud(trainer, opacity_min: float = 0.05) -> torch.Tensor:
    g = trainer.g
    keep = torch.sigmoid(g.opacity_logit) >= opacity_min
    return g.means[keep].detach().cpu()


def run_scene(root: str, scene: str, mode: str, seed: int, epochs: int,
              gt_dir: str) -> Dict[str, float]:
    # train via the suite's run_one, but keep the trainer for the cloud:
    # re-implement the small training loop here to retain the object
    from datasets import EventNeRFDataset
    from render3d import Gaussians3D
    from trainer3d import EvSpikeGSTrainer3D
    from train_real import estimate_background

    ds = EventNeRFDataset(root, scene, n_windows=30)
    torch.manual_seed(seed)
    g = Gaussians3D.random(1500, center=(0, 0, 0), spread=0.3,
                           scale=0.01, seed=seed).to("cpu")
    bg = estimate_background(ds.anchor["image"]) if ds.anchor is not None else 0.0
    theta = 1e-9 if mode == "dense" else THETA[scene]
    trainer = EvSpikeGSTrainer3D(g, backend="torch", theta=theta, mode=mode,
                                 C=ds.C, bg=bg, max_scale=0.05, scene_extent=0.5)
    mask = ds.green_mask.to("cpu")
    n = 0
    for _ in range(epochs):
        for packet in ds.stream():
            kf = packet.keyframe
            if kf is not None:
                kf = {"cam": kf["cam"], "image": kf["image"].to("cpu")}
            trainer.streaming_step_events(packet.cam_prev, packet.cam_curr,
                                          packet.dE.to("cpu"), anchor=kf,
                                          steps=2, pixel_mask=mask)
            n += 1
            if n % 500 == 0:
                trainer.densify_and_prune(max_gaussians=40_000)

    pred = normalise(gaussians_to_cloud(trainer))
    gt = normalise(torch.from_numpy(np.load(os.path.join(gt_dir, f"{scene}.npy"))))
    m = chamfer_distance(pred, gt)
    m.update(scene=scene, mode=mode, seed=seed, n_pred=pred.shape[0])
    del trainer, g
    torch.cuda.empty_cache()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data/nerf")
    ap.add_argument("--gt_dir", type=str, default="data/gt_points")
    ap.add_argument("--scenes", type=str, nargs="+",
                    default=["chair", "drums", "hotdog", "lego", "mic"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--outdir", type=str, default="outputs/real")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows: List[Dict] = []
    print("Chamfer/P2P on EventNeRF-synthetic (scale-normalised, seed "
          f"{args.seed}, {args.epochs} epochs)")
    for scene in args.scenes:
        for mode in ("dense", "lif"):
            try:
                r = run_scene(args.root, scene, mode, args.seed, args.epochs,
                              args.gt_dir)
                rows.append(r)
                print(f"  {scene:<8} {mode:<6}: chamfer {r['chamfer']:.4f} | "
                      f"p2p pred->gt {r['p2p_a2b']:.4f} | gt->pred {r['p2p_b2a']:.4f} | "
                      f"{r['n_pred']} pts", flush=True)
            except Exception as e:
                print(f"  {scene} {mode} FAILED: {e}", flush=True)

    path = os.path.join(args.outdir, "chamfer.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
