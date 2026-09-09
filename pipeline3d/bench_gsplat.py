"""
pipeline3d/bench_gsplat.py  --  streaming-latency benchmark on the REAL CUDA rasterizer.

Measures ms/packet of the full streaming update (render prev+curr, event loss, backward,
LIF gate, Adam step) at DSEC-like resolution with many Gaussians — the scale where the
gsplat backend matters — for dense vs LIF vs LIF+hard-budget configurations.

This substantiates claim C3 on the real backend: the LIF gate + budget bound the update
cost per packet. (Quality is not evaluated here — the event maps are synthetic residual
patterns; use train_synthetic.py / train_real.py for quality.)

Run (Windows, via the env wrapper):
    tools\\run_gpu.bat pipeline3d\\bench_gsplat.py
"""
from __future__ import annotations
import os, sys, argparse
from typing import Dict, List

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from render3d import PinholeCamera, Gaussians3D                    # noqa: E402
from trainer3d import EvSpikeGSTrainer3D                           # noqa: E402
from eval import peak_vram_mb                                      # noqa: E402


def make_cam(H: int, W: int, ang: float) -> PinholeCamera:
    import math
    eye = (1.6 * math.sin(ang), 0.15, 1.6 * math.cos(ang))
    return PinholeCamera.look_at(eye, (0, 0, 0), (0, 1, 0), H, W, fov_deg=60.0)


args_steps = 2       # Adam steps per packet (set from --steps in main)


def bench(cfg_name: str, mode: str, theta: float, budget, n_gauss: int,
          H: int, W: int, packets: int, device: str,
          cull: bool = False) -> Dict[str, float]:
    torch.manual_seed(0)
    g = Gaussians3D.random(n_gauss, center=(0, 0, 0), spread=1.0,
                           scale=0.03, seed=0).to(device)
    trainer = EvSpikeGSTrainer3D(g, backend="gsplat", lr=0.005,
                                 theta=theta, mode=mode, budget=budget, C=0.2,
                                 cull=cull)
    torch.cuda.reset_peak_memory_stats()
    ms: List[float] = []
    fire: List[float] = []
    for k in range(packets + 3):                     # +3 warmup
        cam_prev = make_cam(H, W, 0.02 * k)
        cam_curr = make_cam(H, W, 0.02 * (k + 1))
        # synthetic event map with realistic sparsity (~5% of pixels active)
        dE = torch.zeros(H * W, device=device)
        idx = torch.randperm(H * W, device=device)[: (H * W) // 20]
        dE[idx] = 0.2 * torch.sign(torch.randn(idx.shape[0], device=device))
        f, _ = trainer.streaming_step_events(cam_prev, cam_curr, dE, steps=args_steps)
        if k >= 3:
            ms.append(trainer.last_step_ms)
            fire.append(f)
    return {"config": cfg_name,
            "ms_mean": sum(ms) / len(ms),
            "ms_max": max(ms),
            "active_pct": 100 * sum(fire) / len(fire),
            "vram_mb": peak_vram_mb() or 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_gauss", type=int, default=30_000)
    ap.add_argument("--H", type=int, default=180)
    ap.add_argument("--W", type=int, default=240)
    ap.add_argument("--packets", type=int, default=30)
    ap.add_argument("--theta", type=float, default=0.003)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--steps", type=int, default=2,
                    help="Adam steps per packet; culling amortises the cache over these")
    ap.add_argument("--scaling", action="store_true",
                    help="sweep model size x steps to find where culling starts to pay")
    ap.add_argument("--budget-scaling", action="store_true",
                    help="Prop.4 (B1): fix B, sweep N -> LIF+B cost ~flat in N while dense grows")
    ap.add_argument("--budget", type=int, default=2000, help="hard cap B for --budget-scaling")
    ap.add_argument("--Ns", type=str, default="30000,100000,300000,1000000",
                    help="comma-separated N values for --budget-scaling")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("bench_gsplat.py needs a CUDA GPU")
    global args_steps
    args_steps = args.steps

    if args.scaling:
        print(f"culling scaling study | {args.W}x{args.H} | "
              f"{torch.cuda.get_device_name(0)}")
        print(f"  {'N':>8} {'steps':>6} {'dense ms':>9} {'culled ms':>10} {'speed-up':>9}")
        for n in (30_000, 100_000, 300_000):
            for st in (2, 8):
                args_steps = st
                d = bench("dense", "dense", args.theta, None, n, args.H, args.W,
                          max(8, args.packets // 3), args.device, cull=False)
                c = bench("culled", "lif", args.theta, 500, n, args.H, args.W,
                          max(8, args.packets // 3), args.device, cull=True)
                print(f"  {n:>8} {st:>6} {d['ms_mean']:>9.1f} {c['ms_mean']:>10.1f} "
                      f"{d['ms_mean']/c['ms_mean']:>8.2f}x", flush=True)
        return

    if args.budget_scaling:
        B = args.budget
        Ns = tuple(int(x) for x in args.Ns.split(","))
        print(f"Prop.4 scale test (B1) | fix B={B} | steps={args_steps} | {args.W}x{args.H} | "
              f"{torch.cuda.get_device_name(0)}")
        print(f"  {'N':>9} {'dense ms':>9} {'LIF+B ms':>9} {'LIF vs N0':>10} "
              f"{'active%':>8} {'VRAM MB':>8}")
        rows, base = [], None
        for n in Ns:
            d = bench("dense", "dense", args.theta, None, n, args.H, args.W,
                      args.packets, args.device, cull=False)
            c = bench("lifB", "lif", args.theta, B, n, args.H, args.W,
                      args.packets, args.device, cull=True)
            base = base or c["ms_mean"]
            rows.append((n, d["ms_mean"], c["ms_mean"], c["ms_mean"] / base,
                         c["active_pct"], c["vram_mb"]))
            print(f"  {n:>9} {d['ms_mean']:>9.1f} {c['ms_mean']:>9.1f} "
                  f"{c['ms_mean']/base:>9.2f}x {c['active_pct']:>8.2f} {c['vram_mb']:>8.0f}",
                  flush=True)
        if len(rows) >= 2:
            n0, nL = rows[0][0], rows[-1][0]
            print(f"\n  Prop.4 empirical: N x{nL/n0:.0f} ({n0}->{nL}) => "
                  f"LIF+B cost x{rows[-1][3]:.2f} (bounded), "
                  f"dense x{rows[-1][1]/rows[0][1]:.2f} (grows with N).")
        # B1 auto-save (scale.md §6): one-shot CSV so the lab run needs no manual copy.
        # Writes to gitignored src/outputs/; curate to results_csv/scale.csv when final.
        import csv as _csv
        out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                           "outputs", "scale.csv"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow([f"# GPU={torch.cuda.get_device_name(0)}", f"B={B}",
                        f"steps={args_steps}", f"res={args.W}x{args.H}"])
            w.writerow(["N", "dense_ms", "lifB_ms", "lifB_vs_N0", "active_pct", "vram_mb"])
            for (n, dms, cms, ratio, act, vram) in rows:
                w.writerow([n, f"{dms:.2f}", f"{cms:.2f}", f"{ratio:.3f}",
                            f"{act:.2f}", f"{vram:.0f}"])
        print(f"  [saved] {out}")
        return

    print(f"gsplat streaming benchmark | {args.W}x{args.H}, {args.n_gauss} Gaussians, "
          f"{args.packets} packets | {torch.cuda.get_device_name(0)}")
    configs = [
        ("dense",              "dense", args.theta, None, False),
        ("lif (no cull)",      "lif",   args.theta, None, False),
        ("lif + B=2000",       "lif",   args.theta, 2000, False),
        ("lif + B=500",        "lif",   args.theta, 500,  False),
        ("lif CULLED",         "lif",   args.theta, None, True),
        ("lif CULLED B=2000",  "lif",   args.theta, 2000, True),
        ("lif CULLED B=500",   "lif",   args.theta, 500,  True),
    ]
    print(f"  {'config':<20} {'ms/packet':>10} {'ms max':>8} {'active %':>9} {'VRAM MB':>8}")
    rows = []
    for name, mode, theta, budget, cull in configs:
        r = bench(name, mode, theta, budget, args.n_gauss,
                  args.H, args.W, args.packets, args.device, cull=cull)
        rows.append(r)
        print(f"  {r['config']:<20} {r['ms_mean']:>10.1f} {r['ms_max']:>8.1f} "
              f"{r['active_pct']:>9.1f} {r['vram_mb']:>8.0f}", flush=True)

    base = rows[0]["ms_mean"]
    print("\nspeed-up vs dense:")
    for r in rows[1:]:
        print(f"  {r['config']:<20} {base / r['ms_mean']:.2f}x")
    print("\n'CULLED' removes non-firing Gaussians from the forward AND backward pass "
          "(render_gsplat_subset over a cached inactive layer); the others gate only the "
          "parameter update, so their cost still scales with the full model.")


if __name__ == "__main__":
    main()
