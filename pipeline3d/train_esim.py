"""
train_esim.py -- the "ESIM tier": controlled simulation with a REAL asynchronous event
stream (per-event timestamps, reference-level sensor model — see evspikegs/esim_sim.py),
instead of the window-quantised shortcut of train_synthetic.py.

Renders the ground-truth 3D scene densely along the camera trajectory (sub-frame rate),
simulates an ESIM-style event stream from the log-intensity video, cuts it into packets,
and trains the streaming EvSpike-GS trainer through the SAME packet path used for real
datasets. Ground truth is available by construction, so this tier supports controlled
ablations with a faithful sensor model.

Run:  python pipeline3d/train_esim.py            # CPU
      tools\\run_gpu.bat pipeline3d\\train_esim.py --device cuda
"""
from __future__ import annotations
import os, sys, argparse
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import Gaussians3D, render_torch                     # noqa: E402
from trainer3d import EvSpikeGSTrainer3D                           # noqa: E402
from train_synthetic import build_gt, jitter_cam                   # noqa: E402
from evspikegs.esim_sim import simulate_events, events_to_window_maps  # noqa: E402
from evspikegs.metrics import psnr                                 # noqa: E402

torch.manual_seed(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--n_gauss", type=int, default=120)
    ap.add_argument("--frames", type=int, default=120, help="dense sub-frames simulated")
    ap.add_argument("--windows", type=int, default=24, help="packets per pass")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--C", type=float, default=0.10)
    ap.add_argument("--theta", type=float, default=0.04)
    ap.add_argument("--mode", type=str, default="lif",
                    choices=["lif", "dense", "binary", "topk"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    H, W = args.H, args.W
    gt = build_gt().to(args.device)
    cams = [jitter_cam(i, args.frames, H, W) for i in range(args.frames)]
    with torch.no_grad():
        frames = torch.stack([render_torch(gt, c)[0] for c in cams])       # (T,H,W)
        log_frames = torch.log(frames + 1e-3).cpu()
    frame_ts = torch.linspace(0.0, 1.0, args.frames)

    x, y, t, p = simulate_events(log_frames, frame_ts, C=args.C)
    print(f"ESIM-style stream: {x.numel()} events from {args.frames} sub-frames "
          f"({H}x{W}, C={args.C})")
    maps, edges = events_to_window_maps(x, y, t, p, H, W, args.windows, args.C)

    # camera at each window edge = trajectory interpolated at that time
    def cam_at(tn: float):
        return jitter_cam(tn * (args.frames - 1), args.frames, H, W)

    model = Gaussians3D.random(args.n_gauss, center=(0, 0, 0), spread=0.9,
                               scale=0.10, seed=args.seed).to(args.device)
    trainer = EvSpikeGSTrainer3D(model, backend="torch", lr=0.02,
                                 theta=args.theta, mode=args.mode, C=args.C)
    anchor = {"cam": cams[0], "image": frames[0].to(args.device)}

    print(f"training | mode={args.mode} theta={args.theta} | {args.windows} packets/pass")
    fire_hist = []
    for ep in range(args.epochs):
        for k in range(args.windows):
            f, _ = trainer.streaming_step_events(
                cam_at(float(edges[k])), cam_at(float(edges[k + 1])),
                maps[k].to(args.device),
                anchor=(anchor if k == 0 else None), steps=2)
            fire_hist.append(f)
        with torch.no_grad():
            ev = args.frames // 2
            pq = psnr(trainer.render(trainer.g, cams[ev])[0], frames[ev].to(args.device))
        print(f"  epoch {ep+1}/{args.epochs}: eval PSNR={pq:5.2f} dB | "
              f"avg active-set={100*sum(fire_hist)/len(fire_hist):5.1f}%")

    print(f"\nFinal (ESIM tier): PSNR={pq:.2f} dB, "
          f"active-set={100*sum(fire_hist)/len(fire_hist):.1f}% (mode={args.mode})")


if __name__ == "__main__":
    main()
