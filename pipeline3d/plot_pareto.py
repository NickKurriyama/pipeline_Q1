"""
pipeline3d/plot_pareto.py  --  the quality-efficiency Pareto curve over the firing
threshold theta, on the REAL 3D pipeline (CPU synthetic scene).

Sweeps theta (default {0, 0.01, 0.02, 0.04, 0.08, 0.16} — the paper protocol set),
trains the 3D streaming trainer for each value, and plots:
    (a) PSNR vs active-set sparsity  (b) PSNR vs optimization ms/packet
theta=0 is the dense baseline (gate open). Results are written to CSV so the figure can
be regenerated without retraining (`--plot-only`).

CPU note: numbers here characterize the *mechanism* on the synthetic scene; paper numbers
come from the gsplat backend on EventNeRF/DSEC (GPU — see train_real.py).

Run:  python pipeline3d/plot_pareto.py            # sweep + plot
      python pipeline3d/plot_pareto.py --plot-only  # re-plot from outputs/pareto3d.csv
"""
from __future__ import annotations
import os, sys, csv, argparse
from typing import Dict, List

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import Gaussians3D, render_torch                    # noqa: E402
from trainer3d import EvSpikeGSTrainer3D                          # noqa: E402
from train_synthetic import build_gt, jitter_cam                  # noqa: E402
from evspikegs.metrics import psnr                                # noqa: E402
from eval import EfficiencyMeter                                  # noqa: E402

DEFAULT_THETAS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]


def run_one(theta: float, H: int, W: int, n_gauss: int, views: int, epochs: int,
            seed: int, C: float, device: str = "cpu",
            backend: str = "torch") -> Dict[str, float]:
    """Train the 3D streaming trainer at one theta; return quality + efficiency numbers."""
    torch.manual_seed(seed)
    gt = build_gt().to(device)
    cams = [jitter_cam(i, views, H, W) for i in range(views)]
    with torch.no_grad():
        gt_imgs = [render_torch(gt, c)[0] for c in cams]

    model = Gaussians3D.random(n_gauss, center=(0, 0, 0), spread=0.9,
                               scale=0.10, seed=seed).to(device)
    mode = "dense" if theta <= 0.0 else "lif"
    trainer = EvSpikeGSTrainer3D(model, backend=backend, lr=0.02,
                                 theta=max(theta, 1e-9), mode=mode, C=C)
    anchor = {"cam": cams[0], "image": gt_imgs[0]}

    meter = EfficiencyMeter()
    psnr_hist: List[float] = []
    for ep in range(epochs):
        for k in range(1, views):
            fire, n_upd = trainer.streaming_step(
                cams[k - 1], cams[k], gt_imgs[k - 1], gt_imgs[k],
                anchor=(anchor if k == 1 else None), steps=2)
            meter.update(fire, n_upd, trainer.last_step_ms)
        with torch.no_grad():
            ev = views // 2
            psnr_hist.append(psnr(trainer.render(trainer.g, cams[ev])[0], gt_imgs[ev]))

    eff = meter.summary()
    return {"theta": theta, "psnr": psnr_hist[-1],
            "sparsity_pct": eff["sparsity_pct"], "ms_per_packet": eff["ms_per_packet"]}


def plot(rows: List[Dict[str, float]], out_png: str, backend: str = "torch") -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["theta"])
    dense = next((r for r in rows if r["theta"] == 0.0), None)
    lif = [r for r in rows if r["theta"] > 0.0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, xkey, xlabel in ((a1, "sparsity_pct", "active-set (% Gaussians / window)"),
                             (a2, "ms_per_packet", "optimization ms / packet")):
        xs = [r[xkey] for r in lif]; ys = [r["psnr"] for r in lif]
        ax.plot(xs, ys, "o-", color="tab:blue", label="EvSpike-GS (LIF)")
        for r in lif:
            ax.annotate(f"θ={r['theta']}", (r[xkey], r["psnr"]),
                        textcoords="offset points", xytext=(5, -4), fontsize=8)
        if dense is not None:
            ax.plot([dense[xkey]], [dense["psnr"]], "*", ms=14, color="tab:red",
                    label="dense (θ=0)")
            ax.axhline(dense["psnr"], ls="--", lw=1, color="tab:red", alpha=0.6)
        ax.set_xlabel(xlabel); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=0.3); ax.legend()
    a1.set_title("quality vs update sparsity")
    a2.set_title("quality vs per-packet latency")
    label = ("gsplat CUDA rasterizer" if backend == "gsplat"
             else "pure-torch rasterizer")
    fig.suptitle(f"3D pipeline Pareto over firing threshold ({label}, synthetic scene)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"  wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thetas", type=float, nargs="+", default=DEFAULT_THETAS)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--n_gauss", type=int, default=120)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--C", type=float, default=0.10)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--backend", type=str, default="torch", choices=["torch", "gsplat"],
                    help="gsplat: grad-free projection stimulus -> use ~10x smaller thetas")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--plot-only", action="store_true",
                    help="skip training; re-plot from the existing CSV")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    suffix = "" if args.backend == "torch" else f"_{args.backend}"
    csv_path = os.path.join(args.outdir, f"pareto3d{suffix}.csv")
    png_path = os.path.join(args.outdir, f"pareto3d{suffix}.png")

    if args.plot_only:
        with open(csv_path, newline="") as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
        plot(rows, png_path, backend=args.backend)
        return

    print(f"3D Pareto sweep | thetas={args.thetas} | "
          f"{args.H}x{args.W}, {args.n_gauss} Gaussians, {args.views} views")
    rows: List[Dict[str, float]] = []
    for th in args.thetas:
        r = run_one(th, args.H, args.W, args.n_gauss, args.views,
                    args.epochs, args.seed, args.C, device=args.device,
                    backend=args.backend)
        print(f"  theta={th:<5}: PSNR {r['psnr']:5.2f} dB | "
              f"active {r['sparsity_pct']:5.1f}% | {r['ms_per_packet']:6.1f} ms/packet")
        rows.append(r)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {csv_path}")
    plot(rows, png_path, backend=args.backend)


if __name__ == "__main__":
    main()
