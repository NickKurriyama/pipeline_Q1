"""
pipeline3d/render_images.py -- Train (dung chinh co che nhu train_real.py /
eval_full_gaussian.py) roi LUU ANH RENDER ra file PNG cho TUNG VIEW TEST cua
tung scene / cau hinh, de xem bang mat (khong chi metric so).

Cau truc luu anh (moi scene = 1 folder, moi view = 1 folder con):
    outputs/real/images/<scene>/<view>/gt.png
    outputs/real/images/<scene>/<view>/pred_<tag>.png

Vi du chay 1 scene, 1 cau hinh:
    tools\\run_gpu.bat pipeline3d\\render_images.py --scenes chair --modes lif --theta 1e-5 --tag theta1e-5

Chay het 5 scene, 5 cau hinh (moi lan doi --tag de phan biet):
    tools\\run_gpu.bat pipeline3d\\render_images.py --scenes chair drums hotdog lego mic --modes lif --theta 1e-5 --tag theta1e-5
    tools\\run_gpu.bat pipeline3d\\render_images.py --scenes chair drums hotdog lego mic --modes lif --theta 1e-4 --tag theta1e-4
    tools\\run_gpu.bat pipeline3d\\render_images.py --scenes chair drums hotdog lego mic --modes lif --theta 1e-5 --budget 15000 --tag budget15k
    tools\\run_gpu.bat pipeline3d\\render_images.py --scenes chair drums hotdog lego mic --modes lif --theta 1e-5 --lr 0.01 --n_gauss 30000 --tag lr01_ngauss30k
    tools\\run_gpu.bat pipeline3d\\render_images.py --scenes chair drums hotdog lego mic --modes lif --theta 1e-5 --epochs 10 --n_windows 500 --tag epoch10_window500

Sau khi chay het 5 lenh tren, moi scene se co day du:
    outputs/real/images/chair/r_00250/gt.png
    outputs/real/images/chair/r_00250/pred_theta1e-5.png
    outputs/real/images/chair/r_00250/pred_theta1e-4.png
    outputs/real/images/chair/r_00250/pred_budget15k.png
    outputs/real/images/chair/r_00250/pred_lr01_ngauss30k.png
    outputs/real/images/chair/r_00250/pred_epoch10_window500.png
    ... (tuong tu cho r_00375, r_00500, ... va cho drums/hotdog/lego/mic)
"""
from __future__ import annotations
import os, sys, glob, argparse
from typing import Optional

import numpy as np
import torch
import imageio

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import PinholeCamera, Gaussians3D          # noqa: E402
from trainer3d import EvSpikeGSTrainer3D                  # noqa: E402
from datasets import EventNeRFDataset                     # noqa: E402
from train_real import estimate_background                # noqa: E402

SCENES = ["chair", "drums", "hotdog", "lego", "mic"]
LIF_THETA_DEFAULT = 1e-5


def load_all_test_views(root: str, scene: str, ds: EventNeRFDataset):
    """Lay TOAN BO view trong test/ (hoac validation/), tra ve list (view_name, cam, gt_img)."""
    tdir = os.path.join(root, scene, "test")
    if not os.path.isdir(os.path.join(tdir, "pose")):
        tdir = os.path.join(root, scene, "validation")
    import matplotlib.image as mpimg
    views = []
    for pf in sorted(glob.glob(os.path.join(tdir, "pose", "r_*.txt"))):
        idx = os.path.basename(pf)[2:-4]
        view_name = f"r_{idx}"
        rf = os.path.join(tdir, "rgb", f"r_{idx}.png")
        if not os.path.exists(rf):
            continue
        m = np.loadtxt(pf).reshape(4, 4)
        R_w2c = m[:3, :3].T
        t_w2c = -R_w2c @ m[:3, 3]
        cam = PinholeCamera(ds.H, ds.W, ds.fx, ds.fy, ds.cx, ds.cy,
                            torch.tensor(R_w2c, dtype=torch.float32),
                            torch.tensor(t_w2c, dtype=torch.float32))
        gt_img = mpimg.imread(rf)
        if gt_img.ndim == 3:
            gt_img = gt_img[..., :3].mean(-1)
        views.append((view_name, cam, gt_img))
    return views


def save_png(arr01: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = arr01.detach().cpu().clamp(0, 1).numpy()
    a8 = (a * 255).astype("uint8")
    imageio.imwrite(path, a8)


def run_one(root: str, scene: str, mode: str, seed: int,
            theta: float, budget: Optional[int], lr: float, lam: float,
            steps: int, n_gauss: int, spread: float, max_scale: float,
            n_windows: int, densify_every: int, epochs: int, device: str,
            images_root: str, tag: str,
            anchor_every: int = 25, anchor_weight: float = 0.3,
            anchor_ungated: bool = False):
    ds = EventNeRFDataset(root, scene, n_windows=n_windows, anchor_every=anchor_every)
    torch.manual_seed(seed)
    g = Gaussians3D.random(n_gauss, center=(0, 0, 0), spread=spread,
                           scale=0.01, seed=seed).to(device)
    bg = estimate_background(ds.anchor["image"]) if ds.anchor is not None else 0.0
    trainer = EvSpikeGSTrainer3D(g, backend="gsplat", lr=lr, lam=lam, theta=theta,
                                 mode=mode, C=ds.C, budget=budget, bg=bg,
                                 max_scale=max_scale, scene_extent=spread,
                                 anchor_weight=anchor_weight, anchor_ungated=anchor_ungated)
    mask = ds.green_mask.to(device)

    packet_i = 0
    for _ in range(epochs):
        for packet in ds.stream():
            kf = packet.keyframe
            if kf is not None:
                kf = {"cam": kf["cam"], "image": kf["image"].to(device)}
            trainer.streaming_step_events(
                packet.cam_prev, packet.cam_curr, packet.dE.to(device),
                anchor=kf, steps=steps, pixel_mask=mask)
            packet_i += 1
            if densify_every and packet_i % densify_every == 0:
                trainer.densify_and_prune(max_gaussians=2 * n_gauss)

    views = load_all_test_views(root, scene, ds)
    if not views:
        print(f"[{scene}] khong tim thay test view -- bo qua luu anh")
    else:
        with torch.no_grad():
            for view_name, cam, gt_img in views:
                view_dir = os.path.join(images_root, scene, view_name)
                os.makedirs(view_dir, exist_ok=True)

                gt_path = os.path.join(view_dir, "gt.png")
                if not os.path.exists(gt_path):  # GT giong nhau moi config, luu 1 lan
                    save_png(torch.from_numpy(np.ascontiguousarray(gt_img)).float(), gt_path)

                img, _, _ = trainer.render(trainer.g, cam)
                pred_path = os.path.join(view_dir, f"pred_{tag}.png")
                save_png(img, pred_path)

        print(f"[{scene}] da luu {len(views)} view vao {os.path.join(images_root, scene)}"
              f" (gt.png + pred_{tag}.png moi view)")

    del trainer, g
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data/nerf")
    ap.add_argument("--scenes", type=str, nargs="+", default=SCENES)
    ap.add_argument("--modes", type=str, nargs="+", default=["lif"],
                    choices=["dense", "lif", "binary", "topk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--n_gauss", type=int, default=20_000)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--max_scale", type=float, default=0.05)
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--densify_every", type=int, default=500)

    ap.add_argument("--anchor_every", type=int, default=25,
                    help="Neo lai brightness tuyet doi moi N cua so (mac dinh 25). "
                         "Giam xuong (vd 10) de bot troi dat/nhoe, doi lai train cham hon.")
    ap.add_argument("--anchor_weight", type=float, default=0.3,
                    help="Trong so cua anchor loss trong tong loss (mac dinh 0.3). "
                         "Tang len (vd 0.6) de bam sat anh neo hon, giam troi dat.")
    ap.add_argument("--anchor_ungated", action="store_true",
                    help="Tach anchor loss thanh 1 buoc cap nhat RIENG, KHONG qua LIF "
                         "gate -- sua duoc vung nen/Gaussian it fire ma anchor thuong "
                         "khong voi toi duoc.")

    ap.add_argument("--tag", type=str, default="default",
                    help="Ten config, dung lam ten file pred_<tag>.png trong moi view")
    ap.add_argument("--outdir", type=str, default="outputs/real/images",
                    help="Thu muc goc; ben trong se la <outdir>/<scene>/<view>/")
    args = ap.parse_args()

    print(f"Anh se duoc luu vao: {args.outdir}/<scene>/<view>/\n"
          f"  scenes={args.scenes} modes={args.modes} theta={args.theta} "
          f"budget={args.budget} n_gauss={args.n_gauss} epochs={args.epochs} "
          f"n_windows={args.n_windows} tag={args.tag}", flush=True)

    for scene in args.scenes:
        for mode in args.modes:
            theta = args.theta
            if theta is None:
                theta = 0.0 if mode == "dense" else LIF_THETA_DEFAULT
            try:
                run_one(args.root, scene, mode, args.seed,
                        theta=theta, budget=args.budget, lr=args.lr, lam=args.lam,
                        steps=args.steps, n_gauss=args.n_gauss, spread=args.spread,
                        max_scale=args.max_scale, n_windows=args.n_windows,
                        densify_every=args.densify_every, epochs=args.epochs,
                        device=args.device, images_root=args.outdir, tag=args.tag,
                        anchor_every=args.anchor_every, anchor_weight=args.anchor_weight,
                        anchor_ungated=args.anchor_ungated)
            except Exception as e:
                print(f"[{scene} {mode}] LOI, bo qua: {e}")

    print(f"\nHoan tat. Xem anh trong: {args.outdir}/<scene>/<view>/")


if __name__ == "__main__":
    main()