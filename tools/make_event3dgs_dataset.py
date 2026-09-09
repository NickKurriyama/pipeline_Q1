"""
tools/make_event3dgs_dataset.py -- convert an EventNeRF scene into the dataset layout the
external baseline Event-3DGS (lanpokn/Event-3DGS) expects, so both methods are trained and
evaluated on IDENTICAL data and poses.

Event-3DGS wants
    <out>/images/        intensity images estimated from events
    <out>/images_event/  intensity images used for the difference term
    <out>/renders/       RGB ground truth (evaluation only)
    <out>/sparse/0/      COLMAP text model (cameras.txt, images.txt, points3D.txt)

EventNeRF ships exactly the first ingredient: its `train_e2vid/` split contains E2VID
intensity reconstructions with matching `pose/` (camera-to-world 4x4) and `intrinsics/`.
We reuse those for both `images/` and `images_event/` (the repo notes the two may come from
different estimators; using one estimator for both is the neutral choice), the RGB `train/`
split for `renders/`, and convert the poses to COLMAP's world-to-camera quaternion form.

points3D.txt is seeded with a random cloud inside the normalised object volume, matching the
initialisation our method is given, so neither method gets a geometry advantage.

Usage:
    python tools/make_event3dgs_dataset.py data/data/nerf chair baselines/data/chair
"""
from __future__ import annotations
import os, sys, shutil, glob

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


def main(root: str, scene: str, out: str, n_points: int = 20_000) -> None:
    src_e2v = os.path.join(root, scene, "train_e2vid")
    src_rgb = os.path.join(root, scene, "train", "rgb")
    for d in ("images", "images_event", "renders", os.path.join("sparse", "0")):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    # ---- intensity images (events) -> images/ and images_event/ ----------------------
    imgs = sorted(glob.glob(os.path.join(src_e2v, "rgb", "*.png")))
    if not imgs:
        raise SystemExit(f"no e2vid images under {src_e2v}/rgb")
    names = []
    for p in imgs:
        n = os.path.basename(p)
        names.append(n)
        shutil.copyfile(p, os.path.join(out, "images", n))
        shutil.copyfile(p, os.path.join(out, "images_event", n))

    # ---- RGB ground truth -> renders/ (nearest frame by index) -----------------------
    rgbs = sorted(glob.glob(os.path.join(src_rgb, "*.png")))
    if rgbs:
        # e2vid frames are uniformly spaced over the same rotation as the RGB split
        for k, n in enumerate(names):
            j = min(len(rgbs) - 1, round(k * (len(rgbs) - 1) / max(len(names) - 1, 1)))
            shutil.copyfile(rgbs[j], os.path.join(out, "renders", n))

    # ---- intrinsics ------------------------------------------------------------------
    intr = sorted(glob.glob(os.path.join(src_e2v, "intrinsics", "*.txt")))
    K = np.loadtxt(intr[0]).reshape(-1)[:16].reshape(4, 4)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    W, H = int(round(2 * cx)), int(round(2 * cy))
    with open(os.path.join(out, "sparse", "0", "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # ---- poses: EventNeRF camera-to-world -> COLMAP world-to-camera ------------------
    poses = sorted(glob.glob(os.path.join(src_e2v, "pose", "*.txt")))
    if len(poses) != len(names):
        print(f"warning: {len(poses)} poses vs {len(names)} images; pairing by sort order")
    centers = []
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

    # ---- seed point cloud (same volume our method is initialised in) -----------------
    rng = np.random.default_rng(0)
    pts = 0.5 * (rng.random((n_points, 3)) - 0.5)
    with open(os.path.join(out, "sparse", "0", "points3D.txt"), "w") as f:
        f.write("# 3D point list\n")
        for i, p in enumerate(pts, start=1):
            f.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128 0.0\n")

    print(f"wrote Event-3DGS dataset -> {out}")
    print(f"  {len(names)} images ({W}x{H}, fx={fx:.1f}) | {len(poses)} poses | "
          f"{n_points} seed points")
    print(f"  camera centres radius: {np.linalg.norm(np.array(centers), axis=1).mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
