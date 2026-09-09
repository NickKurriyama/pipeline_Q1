"""
tools/make_evgs_dataset.py -- convert an EventNeRF scene into a NEUTRAL bundle for the external
baseline Ev-GS (Wu et al., MLSP'24), so Ev-GS and EvSpike-GS train/eval on IDENTICAL data,
poses and initialisation. Mirrors tools/make_event3dgs_dataset.py but emits BOTH a COLMAP model
and a NeRF-style transforms.json, plus a grayscale image split (Ev-GS is monocular grayscale),
because we cannot assume which loader the Ev-GS repo expects.

Emits under <out>/:
    rgb/                 grayscale intensity frames (from EventNeRF train_e2vid, L-mode)
    sparse/0/            COLMAP text model (cameras.txt, images.txt, points3D.txt)
    transforms.json      NeRF-style (camera_angle_x + per-frame c2w) -> for NeRF-style loaders
    intrinsics.txt       fx fy cx cy W H
    points3d_seed.ply    random seed cloud in the normalised object volume (same as ours)

>>> ADAPT TO THE EV-GS REPO <<<  Read the Ev-GS README and point its data loader at either the
COLMAP `sparse/0` + `rgb/` split, or `transforms.json`. If Ev-GS ingests RAW events rather than
intensity frames, use EventNeRF's native event files directly and keep only the poses/seed here.
The neutral choice (one estimator's intensity for a grayscale method) matches how we ran the
Event-3DGS baseline; document any deviation in the paper's baseline caveats.

Usage:
    python tools/make_evgs_dataset.py data/data/nerf chair baselines/data/chair_evgs
"""
from __future__ import annotations
import os, sys, glob, json, shutil

import numpy as np


def rot_to_qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) from a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-8:
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4); q[i + 1] = 1.0
        return q
    return np.array([w,
                     (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def to_grayscale(src: str, dst: str) -> None:
    """Copy src->dst as an L-mode PNG if PIL is available, else a plain copy."""
    try:
        from PIL import Image
        Image.open(src).convert("L").save(dst)
    except Exception:
        shutil.copyfile(src, dst)


def write_seed_ply(path: str, pts: np.ndarray) -> None:
    """Minimal ASCII PLY with x,y,z (+ gray colour) for 3DGS initialisation."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128\n")


def main(root: str, scene: str, out: str, n_points: int = 20_000) -> None:
    src_e2v = os.path.join(root, scene, "train_e2vid")
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)

    # ---- grayscale intensity frames -> rgb/ ------------------------------------------
    imgs = sorted(glob.glob(os.path.join(src_e2v, "rgb", "*.png")))
    if not imgs:
        raise SystemExit(f"no e2vid images under {src_e2v}/rgb")
    names = []
    for p in imgs:
        n = os.path.basename(p)
        names.append(n)
        to_grayscale(p, os.path.join(out, "rgb", n))

    # ---- intrinsics ------------------------------------------------------------------
    intr = sorted(glob.glob(os.path.join(src_e2v, "intrinsics", "*.txt")))
    K = np.loadtxt(intr[0]).reshape(-1)[:16].reshape(4, 4)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    W, H = int(round(2 * cx)), int(round(2 * cy))
    with open(os.path.join(out, "intrinsics.txt"), "w") as f:
        f.write(f"{fx} {fy} {cx} {cy} {W} {H}\n")
    with open(os.path.join(out, "sparse", "0", "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # ---- poses: EventNeRF c2w -> COLMAP w2c (images.txt) + NeRF transforms.json -------
    poses = sorted(glob.glob(os.path.join(src_e2v, "pose", "*.txt")))
    if len(poses) != len(names):
        print(f"warning: {len(poses)} poses vs {len(names)} images; pairing by sort order")
    frames, centers = [], []
    with open(os.path.join(out, "sparse", "0", "images.txt"), "w") as f:
        f.write("# Image list, two lines each\n")
        for i, (pf, n) in enumerate(zip(poses, names), start=1):
            c2w = np.loadtxt(pf).reshape(4, 4)
            R_w2c = c2w[:3, :3].T
            t_w2c = -R_w2c @ c2w[:3, 3]
            centers.append(c2w[:3, 3])
            q = rot_to_qvec(R_w2c)
            f.write(f"{i} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {n}\n\n")
            frames.append({"file_path": f"rgb/{n}", "transform_matrix": c2w.tolist()})

    camera_angle_x = float(2.0 * np.arctan(0.5 * W / fx))
    with open(os.path.join(out, "transforms.json"), "w") as f:
        json.dump({"camera_angle_x": camera_angle_x, "fl_x": fx, "fl_y": fy,
                   "cx": cx, "cy": cy, "w": W, "h": H, "frames": frames}, f, indent=1)

    # ---- seed point cloud (same volume our method is initialised in) ------------------
    rng = np.random.default_rng(0)
    pts = 0.5 * (rng.random((n_points, 3)) - 0.5)
    with open(os.path.join(out, "sparse", "0", "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for i, p in enumerate(pts, start=1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128 0.0\n")
    write_seed_ply(os.path.join(out, "points3d_seed.ply"), pts)

    print(f"wrote Ev-GS neutral bundle -> {out}")
    print(f"  {len(names)} gray frames ({W}x{H}, fx={fx:.1f}) | {len(poses)} poses | "
          f"{n_points} seed pts")
    print(f"  camera-centre radius: {np.linalg.norm(np.array(centers), axis=1).mean():.3f}")
    print("  NEXT: point the Ev-GS loader at sparse/0 + rgb/ (COLMAP) or transforms.json;")
    print("        after training, eval the output PLY with pipeline3d/eval_baseline.py --method Ev-GS")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3])
