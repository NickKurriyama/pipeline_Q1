"""
pipeline3d/eval_full_gaussian.py -- Danh gia DAY DU metric cho he thong EvSpikeGS
(Gaussian streaming, khong phai baseline NeRF).

Gop lai tu:
  - train_real.py      : cac tham so training linh hoat (theta, budget, lr, n_gauss,...)
  - eval_real_suite.py : train dense/lif, novel-view PSNR/SSIM tren test/, active%, ms/packet
  - eval_chamfer.py    : Chamfer + P2P giua Gaussian cloud va GT mesh (Blender export)
  - eval.py             : LPIPS (tuy chon, can `pip install lpips`)

Ket qua: LUON append (khong ghi de) vao 1 file CSV DUY NHAT (mac dinh
outputs/real/full_metrics.csv), moi dong = 1 lan chay (scene, mode, seed, theta,...)
voi day du: psnr, ssim, lpips, active_pct, ms_per_packet, chamfer, p2p_a2b, p2p_b2a,
n_pred_points, n_test_views + toan bo hyperparam da dung -- de ban chay nhieu lan
voi tham so khac nhau (sweep theta/budget/lr/...) ma van gom chung 1 bang so sanh.

Chay 1 lan mac dinh (het cac scene, mode dense+lif, seed 0):
    tools\\run_gpu.bat pipeline3d\\eval_full_gaussian.py

Chay sweep nhieu tham so (moi lan goi se APPEND them dong vao cung 1 CSV):
    tools\\run_gpu.bat pipeline3d\\eval_full_gaussian.py --scenes chair --modes lif --theta 1e-5
    tools\\run_gpu.bat pipeline3d\\eval_full_gaussian.py --scenes chair --modes lif --theta 3e-5
    tools\\run_gpu.bat pipeline3d\\eval_full_gaussian.py --scenes chair --modes lif --theta 1e-4 --budget 15000
    tools\\run_gpu.bat pipeline3d\\eval_full_gaussian.py --scenes chair --modes lif --lr 0.01 --n_gauss 30000
    # --> tat ca cac dong tren deu duoc GOM CHUNG vao outputs/real/full_metrics.csv
"""
from __future__ import annotations
import os, sys, csv, glob, argparse, statistics, traceback
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import PinholeCamera, Gaussians3D                    # noqa: E402
from trainer3d import EvSpikeGSTrainer3D                           # noqa: E402
from datasets import EventNeRFDataset                               # noqa: E402
from train_real import estimate_background                         # noqa: E402
from eval import psnr, ssim, chamfer_distance                      # noqa: E402

SCENES = ["chair", "drums", "hotdog", "lego", "mic"]
LIF_THETA_DEFAULT = 1e-5
OPACITY_MIN = 0.05

try:
    import lpips as lpips_lib
    _LPIPS_MODEL = lpips_lib.LPIPS(net="alex")
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False
    print("[canh bao] chua cai package 'lpips' -> cot LPIPS se ghi N/A. "
          "Cai bang: pip install lpips --break-system-packages")


# ------------------------------------------------------------------------------------------
# held-out test views (giong eval_real_suite.py)
# ------------------------------------------------------------------------------------------
def load_test_views(root: str, scene: str, ds: EventNeRFDataset):
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
        views.append((f"r_{idx}", cam, torch.from_numpy(np.ascontiguousarray(img)).float()))
    return views


def lpips_gray(pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    if not _HAS_LPIPS:
        return None
    def to3(x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(1, 1, *x.shape[-2:]).repeat(1, 3, 1, 1) * 2.0) - 1.0
    with torch.no_grad():
        d = _LPIPS_MODEL(to3(pred.float().cpu()), to3(target.float().cpu()))
    return float(d.item())


@torch.no_grad()
def eval_novel_views(trainer, views, device, verbose: bool = False,
                     scene: str = "", mode: str = "") -> Dict[str, float]:
    ps, ss, lp = [], [], []
    for name, cam, ref in views:
        img, _, _ = trainer.render(trainer.g, cam)
        ref = ref.to(img.device)
        p = psnr(img, ref)
        s = ssim(img, ref)
        ps.append(p)
        ss.append(s)
        val = lpips_gray(img, ref)
        if val is not None:
            lp.append(val)
        if verbose:
            lp_str = f" | LPIPS {val:.4f}" if val is not None else ""
            print(f"    [{scene} {mode}] {name}: PSNR {p:.3f} | SSIM {s:.4f}{lp_str}",
                 flush=True)
    out = {"psnr": sum(ps) / len(ps), "ssim": sum(ss) / len(ss)}
    out["lpips"] = (sum(lp) / len(lp)) if lp else "N/A"
    if verbose:
        lp_str = f" | mean LPIPS {out['lpips']:.4f}" if out["lpips"] != "N/A" else ""
        print(f"    [{scene} {mode}] mean PSNR {out['psnr']:.3f} | "
              f"mean SSIM {out['ssim']:.4f}{lp_str}", flush=True)
    return out


# ------------------------------------------------------------------------------------------
# Chamfer/P2P (giong eval_chamfer.py) -- dung chinh trainer da train, khong train lai
# ------------------------------------------------------------------------------------------
def normalise(P: torch.Tensor) -> torch.Tensor:
    P = P - P.mean(dim=0, keepdim=True)
    return P / P.pow(2).sum(dim=1).mean().sqrt().clamp_min(1e-9)


def gaussians_to_cloud(trainer, opacity_min: float = OPACITY_MIN) -> torch.Tensor:
    g = trainer.g
    keep = torch.sigmoid(g.opacity_logit) >= opacity_min
    return g.means[keep].detach().cpu()


def eval_geometry(trainer, scene: str, gt_dir: str, max_points: int = 5000) -> Dict[str, float]:
    gt_path = os.path.join(gt_dir, f"{scene}.npy")
    if not os.path.exists(gt_path):
        return {"chamfer": "N/A", "p2p_a2b": "N/A", "p2p_b2a": "N/A", "n_pred": 0}
    pred = normalise(gaussians_to_cloud(trainer))
    gt = normalise(torch.from_numpy(np.load(gt_path)))
    # subsample de tranh OOM: torch.cdist tao ma tran (chunk x M), voi M lon (vd
    # 100k diem GT tu Blender export) se no bo nho -- gioi han ca 2 phia xuong
    # toi da max_points, du de uoc luong Chamfer on dinh tren CPU.
    g = torch.Generator().manual_seed(0)
    if pred.shape[0] > max_points:
        idx = torch.randperm(pred.shape[0], generator=g)[:max_points]
        pred = pred[idx]
    if gt.shape[0] > max_points:
        idx = torch.randperm(gt.shape[0], generator=g)[:max_points]
        gt = gt[idx]
    m = chamfer_distance(pred, gt)
    m["n_pred"] = pred.shape[0]
    return m


# ------------------------------------------------------------------------------------------
# mot lan train + danh gia day du, nhan TOAN BO hyperparam linh hoat nhu train_real.py
# ------------------------------------------------------------------------------------------
def run_one_full(root: str, scene: str, mode: str, seed: int, gt_dir: str,
                 theta: float, budget: Optional[int], lr: float, lam: float,
                 steps: int, n_gauss: int, spread: float, max_scale: float,
                 n_windows: int, densify_every: int, topk_frac: float,
                 epochs: int, device: str, anchor_every: int = 25,
                 anchor_weight: float = 0.3, anchor_ungated: bool = False,
                 verbose_views: bool = False) -> Dict[str, float]:
    ds = EventNeRFDataset(root, scene, n_windows=n_windows, anchor_every=anchor_every)
    torch.manual_seed(seed)
    g = Gaussians3D.random(n_gauss, center=(0, 0, 0), spread=spread,
                           scale=0.01, seed=seed).to(device)
    bg = estimate_background(ds.anchor["image"]) if ds.anchor is not None else 0.0
    trainer = EvSpikeGSTrainer3D(g, backend="gsplat", lr=lr, lam=lam, theta=theta,
                                 mode=mode, C=ds.C, budget=budget, bg=bg,
                                 max_scale=max_scale, scene_extent=spread,
                                 anchor_weight=anchor_weight, anchor_ungated=anchor_ungated)
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
                anchor=kf, steps=steps, pixel_mask=mask)
            fire_hist.append(f)
            ms_hist.append(trainer.last_step_ms)
            packet_i += 1
            if densify_every and packet_i % densify_every == 0:
                trainer.densify_and_prune(max_gaussians=2 * n_gauss)

    views = load_test_views(root, scene, ds)
    img_m = eval_novel_views(trainer, views, device, verbose=verbose_views,
                             scene=scene, mode=mode)
    geo_m = eval_geometry(trainer, scene, gt_dir)

    row = {
        "run": None,  # duoc gan trong vong lap main(), giu cho o day de FIELDS khop
        "scene": scene, "mode": mode, "seed": seed,
        "theta": theta, "budget": budget if budget is not None else "None",
        "lr": lr, "lam": lam, "steps": steps, "n_gauss": n_gauss,
        "spread": spread, "max_scale": max_scale, "topk_frac": topk_frac,
        "n_windows": n_windows, "densify_every": densify_every,
        "epochs": epochs,
        "psnr": round(img_m["psnr"], 3),
        "ssim": round(img_m["ssim"], 3),
        "lpips": round(img_m["lpips"], 3) if img_m["lpips"] != "N/A" else "N/A",
        "active_pct": round(100 * sum(fire_hist) / len(fire_hist), 2),
        "ms_per_packet": round(sum(ms_hist) / len(ms_hist), 2),
        "chamfer": round(geo_m["chamfer"], 4) if geo_m["chamfer"] != "N/A" else "N/A",
        "p2p_pred_to_gt": round(geo_m["p2p_a2b"], 4) if geo_m["p2p_a2b"] != "N/A" else "N/A",
        "p2p_gt_to_pred": round(geo_m["p2p_b2a"], 4) if geo_m["p2p_b2a"] != "N/A" else "N/A",
        "n_pred_points": geo_m["n_pred"],
        "n_test_views": len(views),
    }
    del trainer, g
    torch.cuda.empty_cache()
    return row


# ------------------------------------------------------------------------------------------
# CSV helper -- LUON append vao 1 file duy nhat, du chay bao nhieu lan voi tham so gi
# ------------------------------------------------------------------------------------------
FIELDS = ["run", "scene", "mode", "seed", "theta", "budget", "lr", "lam", "steps",
          "n_gauss", "spread", "max_scale", "topk_frac", "n_windows",
          "densify_every", "epochs",
          "psnr", "ssim", "lpips", "active_pct", "ms_per_packet",
          "chamfer", "p2p_pred_to_gt", "p2p_gt_to_pred",
          "n_pred_points", "n_test_views"]


def append_csv(path: str, row: Dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in FIELDS})


# ------------------------------------------------------------------------------------------
# main -- tham so linh hoat nhu train_real.py, + scenes/modes/seeds de quet nhieu scene
# ------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/data/nerf")
    ap.add_argument("--gt_dir", type=str, default="data/gt_points")
    ap.add_argument("--scenes", type=str, nargs="+", default=SCENES)
    ap.add_argument("--modes", type=str, nargs="+", default=["dense", "lif"],
                    choices=["dense", "lif", "binary", "topk"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")

    # ---- hyperparam linh hoat, giong train_real.py -- doi tung cai de sweep ----
    ap.add_argument("--theta", type=float, default=None,
                    help="LIF threshold. Neu bo trong: dense=0.0, lif=1e-5 (mac dinh)")
    ap.add_argument("--budget", type=int, default=None, help="hard active-set cap B")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=2, help="Adam steps moi packet")
    ap.add_argument("--n_gauss", type=int, default=20_000)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--max_scale", type=float, default=0.05)
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--densify_every", type=int, default=500)
    ap.add_argument("--topk_frac", type=float, default=0.3)
    ap.add_argument("--anchor_every", type=int, default=25,
                    help="Neo brightness moi X cua so (mac dinh 25)")
    ap.add_argument("--anchor_weight", type=float, default=0.3,
                    help="Trong so anchor_loss (mac dinh 0.3)")
    ap.add_argument("--anchor_ungated", action="store_true",
                    help="Anchor cap nhat rieng, khong qua LIF gate")
    ap.add_argument("--verbose_views", action="store_true",
                    help="In PSNR/SSIM/LPIPS rieng tung test view (de chan doan view "
                         "nao te nhat), ngoai dong trung binh ghi vao CSV")

    ap.add_argument("--outdir", type=str, default="outputs/real")
    ap.add_argument("--csv_name", type=str, default="full_metrics.csv",
                    help="Ten file CSV DUY NHAT cho bo tham so nay -- chua ca "
                         "--repeats lan chay x 5 scene, phan biet bang cot 'run'")
    ap.add_argument("--repeats", type=int, default=4,
                    help="So lan lap lai TOAN BO quy trinh (5 scene) voi CUNG 1 bo "
                         "tham so; tat ca ghi vao CUNG 1 file CSV, cot 'run' phan biet")
    ap.add_argument("--skip_lpips", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, args.csv_name)

    global _HAS_LPIPS
    if args.skip_lpips:
        _HAS_LPIPS = False

    print(f"Danh gia DAY DU tren he EvSpikeGS -> 1 file CSV duy nhat: {csv_path}\n"
          f"  {args.repeats} lan chay x {len(args.scenes)} scene = "
          f"{args.repeats * len(args.scenes) * len(args.modes) * len(args.seeds)} dong\n"
          f"  scenes={args.scenes} modes={args.modes} seeds={args.seeds} "
          f"epochs={args.epochs} LPIPS={'BAT' if _HAS_LPIPS else 'TAT'}\n"
          f"  theta={args.theta} budget={args.budget} lr={args.lr} lam={args.lam} "
          f"n_gauss={args.n_gauss} spread={args.spread} max_scale={args.max_scale}",
          flush=True)

    for run_i in range(1, args.repeats + 1):
        print(f"\n===== RUN {run_i}/{args.repeats} (5 scene) =====", flush=True)

        for scene in args.scenes:
            for mode in args.modes:
                theta = args.theta
                if theta is None:
                    theta = 0.0 if mode == "dense" else LIF_THETA_DEFAULT
                for seed in args.seeds:
                    try:
                        r = run_one_full(
                            args.root, scene, mode, seed, args.gt_dir,
                            theta=theta, budget=args.budget, lr=args.lr, lam=args.lam,
                            steps=args.steps, n_gauss=args.n_gauss, spread=args.spread,
                            max_scale=args.max_scale, n_windows=args.n_windows,
                            densify_every=args.densify_every, topk_frac=args.topk_frac,
                            epochs=args.epochs, device=args.device,
                            anchor_every=args.anchor_every, anchor_weight=args.anchor_weight,
                            anchor_ungated=args.anchor_ungated,
                            verbose_views=args.verbose_views)
                        r["run"] = run_i
                        append_csv(csv_path, r)
                        print(f"  run{run_i} {scene:<8} {mode:<6} seed={seed} "
                              f"theta={theta:g}: PSNR {r['psnr']:>6} | SSIM {r['ssim']:>6} | "
                              f"LPIPS {r['lpips']:>6} | Chamfer {r['chamfer']:>8} | "
                              f"active {r['active_pct']:>5}% | {r['ms_per_packet']:>6} ms/pkt",
                              flush=True)
                    except Exception:
                        print(f"  {scene} {mode} seed={seed} run{run_i} FAILED:\n"
                              f"{traceback.format_exc()}", flush=True)

    print(f"\nHoan tat {args.repeats} lan chay x {len(args.scenes)} scene. "
          f"Toan bo ket qua nam trong 1 file: {csv_path}", flush=True)


if __name__ == "__main__":
    main()