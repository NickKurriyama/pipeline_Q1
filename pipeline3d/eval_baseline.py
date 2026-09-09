"""
pipeline3d/eval_baseline.py -- evaluate the EXTERNAL baseline Event-3DGS (lanpokn) with
OUR metrics on OUR held-out views, so the comparison is head-to-head.

The baseline writes a standard 3DGS point-cloud PLY. We load it, convert the parameters to
our Gaussians3D container and render it with the SAME rasteriser (gsplat) and the SAME
held-out EventNeRF test cameras used for Table "main", then report PSNR/SSIM/LPIPS in
grayscale. This removes every confound except the reconstruction itself: same data, same
poses, same renderer, same metric code.

Note: the baseline optimises SH colour; we evaluate luminance, taking the DC SH term
(band 0) as the view-independent colour, which is what a grayscale comparison needs.

Usage:
  tools\\run_gpu.bat pipeline3d\\eval_baseline.py ^
      --ply ..\\baselines\\out\\chair_e3dgs\\point_cloud\\iteration_8000\\point_cloud.ply ^
      --scene chair
"""
from __future__ import annotations
import os, sys, argparse

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import Gaussians3D, render_gsplat                    # noqa: E402
from datasets import EventNeRFDataset                              # noqa: E402
from eval_real_suite import load_test_views                        # noqa: E402
from evspikegs.metrics import psnr, ssim                           # noqa: E402

SH_C0 = 0.28209479177387814          # band-0 spherical-harmonics constant


def load_3dgs_ply(path: str, device: str = "cuda") -> Gaussians3D:
    """Read a vanilla-3DGS point_cloud.ply into our container (DC colour only)."""
    from plyfile import PlyData
    ply = PlyData.read(path)["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1)
    opacity = np.asarray(ply["opacity"])[:, None]
    scales = np.stack([ply[f"scale_{i}"] for i in range(3)], axis=1)
    rots = np.stack([ply[f"rot_{i}"] for i in range(4)], axis=1)
    dc = np.stack([ply[f"f_dc_{i}"] for i in range(3)], axis=1)
    rgb = np.clip(SH_C0 * dc + 0.5, 0.0, 1.0)                      # SH DC -> [0,1] colour
    lum = rgb.mean(axis=1)                                         # grayscale luminance

    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return Gaussians3D(t(xyz), t(scales), t(rots), t(opacity[:, 0]), t(lum),
                       learnable=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--scene", default="chair")
    ap.add_argument("--method", default="Event-3DGS",
                    help="label for the printout, e.g. 'Ev-GS' (works for any 3DGS-format PLY)")
    ap.add_argument("--root", default="data/data/nerf")
    ap.add_argument("--bg", type=float, default=None,
                    help="background luminance (default: from the anchor border)")
    a = ap.parse_args()

    g = load_3dgs_ply(a.ply)
    print(f"loaded baseline model: {g.n} Gaussians")

    ds = EventNeRFDataset(a.root, a.scene, n_windows=300)
    views = load_test_views(a.root, a.scene, ds)
    bg = a.bg
    if bg is None and ds.anchor is not None:
        img = ds.anchor["image"]
        border = torch.cat([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        bg = float(border.median())
    print(f"held-out views: {len(views)} | background {bg:.3f}")

    ps, ss = [], []
    lp = []
    try:
        from final_table import lpips_value
    except Exception:
        lpips_value = lambda *_: None
    with torch.no_grad():
        for cam, ref in views:
            img, _, _ = render_gsplat(g, cam, bg=bg)
            ref_d = ref.to(img.device)
            ps.append(psnr(img, ref_d)); ss.append(ssim(img, ref_d))
            v = lpips_value(img, ref_d)
            if v is not None:
                lp.append(v)
    n = len(ps)
    print(f"\n{a.method} (external baseline) on {a.scene}, our held-out views, our metrics:")
    print(f"  PSNR  {sum(ps)/n:6.2f} dB")
    print(f"  SSIM  {sum(ss)/n:6.3f}")
    if lp:
        print(f"  LPIPS {sum(lp)/len(lp):6.3f}")
    print(f"  Gaussians {g.n} (updates every one of them per iteration)")


if __name__ == "__main__":
    main()
