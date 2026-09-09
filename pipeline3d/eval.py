"""
pipeline3d/eval.py  --  evaluation metrics + efficiency measurement for the paper protocol.

Metric groups (roadmap section 5.3):
  image      : PSNR / SSIM (CPU, implemented) ; LPIPS (needs `pip install lpips`, GPU box)
  geometry   : depth RMSE / L1 vs LiDAR (DSEC) ; Chamfer + point-to-point (synthetic)
  efficiency : sparsity %, active-set size, ms/packet, peak VRAM (CUDA only)

Everything here runs on CPU except:
  * `lpips_fn`     — optional dependency, meant for the GPU eval box (NOT RUN here).
  * `peak_vram_mb` — returns None without CUDA (NOT RUN here).

Run `python pipeline3d/eval.py` for a CPU self-check on random data.
"""
from __future__ import annotations
import os, sys, time
from typing import Callable, Dict, List, Optional, Tuple

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from evspikegs.metrics import psnr, ssim  # noqa: E402  (re-exported for one-stop import)

__all__ = ["psnr", "ssim", "lpips_fn", "depth_rmse_vs_lidar", "chamfer_distance",
           "measure_ms_per_packet", "peak_vram_mb", "sparsity_pct", "EfficiencyMeter"]


# ------------------------------------------------------------------------------------------
# image (LPIPS is GPU-box only)
# ------------------------------------------------------------------------------------------
def lpips_fn(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> float:
    """
    LPIPS perceptual distance. Requires `pip install lpips` (downloads pretrained weights)
    — intended for the GPU evaluation box; NOT RUN in the CPU-only environment.
    pred/target: (H,W) grayscale in [0,1]; replicated to 3 channels and mapped to [-1,1].
    """
    import lpips  # deferred: optional dependency
    model = lpips.LPIPS(net=net)
    def to3(x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(1, 1, *x.shape[-2:]).repeat(1, 3, 1, 1) * 2.0) - 1.0
    with torch.no_grad():
        return float(model(to3(pred.float()), to3(target.float())).item())


# ------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------
def depth_rmse_vs_lidar(depth_pred: torch.Tensor, depth_lidar: torch.Tensor,
                        valid: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Depth error against (sparse) LiDAR ground truth — the geometry metric for DSEC.
    depth_pred/depth_lidar: (H,W); valid: bool mask of pixels with a LiDAR return
    (defaults to depth_lidar > 0, the standard sparse-projection convention).
    Returns {'rmse': ..., 'l1': ..., 'n_valid': ...}.
    """
    if valid is None:
        valid = depth_lidar > 0
    d = (depth_pred[valid] - depth_lidar[valid]).float()
    if d.numel() == 0:
        return {"rmse": float("nan"), "l1": float("nan"), "n_valid": 0}
    return {"rmse": float(torch.sqrt((d ** 2).mean()).item()),
            "l1": float(d.abs().mean().item()),
            "n_valid": int(d.numel())}


def chamfer_distance(a: torch.Tensor, b: torch.Tensor,
                     chunk: int = 4096) -> Dict[str, float]:
    """
    Symmetric Chamfer distance + per-direction point-to-point means between two point
    clouds a (N,3) and b (M,3). Only valid on synthetic scenes with clean GT geometry
    (EventNeRF-synthetic / ESIM) — do NOT report on unbounded outdoor scenes.
    Chunked O(N*M) — fine for <=1e5 points on CPU.
    """
    def nn_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out: List[torch.Tensor] = []
        for i in range(0, x.shape[0], chunk):
            d2 = torch.cdist(x[i:i + chunk], y)          # (c, M)
            out.append(d2.min(dim=1).values)
        return torch.cat(out)
    a2b = nn_dist(a.float(), b.float())
    b2a = nn_dist(b.float(), a.float())
    return {"chamfer": float((a2b.mean() + b2a.mean()).item()),
            "p2p_a2b": float(a2b.mean().item()),
            "p2p_b2a": float(b2a.mean().item())}


# ------------------------------------------------------------------------------------------
# efficiency
# ------------------------------------------------------------------------------------------
def sparsity_pct(fire_history: List[float]) -> float:
    """Average firing fraction over a run, in percent."""
    return 100.0 * sum(fire_history) / max(1, len(fire_history))


def measure_ms_per_packet(step_fn: Callable[[], None], n_packets: int = 20,
                          warmup: int = 3) -> Dict[str, float]:
    """
    Wall-clock latency of a streaming update. `step_fn` performs ONE full packet update
    (render active set + loss + backward + optimizer step). Reports mean/std/max ms.
    On CUDA the caller must synchronize inside step_fn for honest numbers.
    """
    for _ in range(warmup):
        step_fn()
    times: List[float] = []
    for _ in range(n_packets):
        t0 = time.perf_counter()
        step_fn()
        times.append((time.perf_counter() - t0) * 1e3)
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / max(1, len(times) - 1)
    return {"ms_mean": mean, "ms_std": var ** 0.5, "ms_max": max(times)}


def peak_vram_mb(device: Optional[torch.device] = None) -> Optional[float]:
    """Peak allocated VRAM in MB since the last reset. None without CUDA (CPU-only env)."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


class EfficiencyMeter:
    """Accumulates per-packet efficiency stats during a streaming run."""

    def __init__(self) -> None:
        self.fire: List[float] = []
        self.ms: List[float] = []
        self.active: List[int] = []

    def update(self, fire_frac: float, n_active: int, step_ms: float) -> None:
        self.fire.append(fire_frac)
        self.active.append(n_active)
        self.ms.append(step_ms)

    def summary(self) -> Dict[str, float]:
        n = max(1, len(self.ms))
        out = {"sparsity_pct": sparsity_pct(self.fire),
               "active_mean": sum(self.active) / n,
               "ms_per_packet": sum(self.ms) / n}
        vram = peak_vram_mb()
        if vram is not None:
            out["peak_vram_mb"] = vram
        return out


# ------------------------------------------------------------------------------------------
# CPU self-check
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    img = torch.rand(32, 32)
    noisy = (img + 0.05 * torch.randn(32, 32)).clamp(0, 1)
    print(f"psnr(img, noisy)   = {psnr(noisy, img):.2f} dB")
    print(f"ssim(img, noisy)   = {ssim(noisy, img):.3f}")
    print(f"psnr(img, img)     = {psnr(img, img):.1f} dB (should be huge)")

    depth_gt = torch.zeros(32, 32); depth_gt[::4, ::4] = 5.0          # sparse "LiDAR"
    depth_pred = torch.full((32, 32), 5.2)
    print("depth vs lidar     =", depth_rmse_vs_lidar(depth_pred, depth_gt))

    a = torch.randn(500, 3); b = a + 0.01 * torch.randn(500, 3)
    print("chamfer (a, a+eps) =", {k: round(v, 4) for k, v in chamfer_distance(a, b).items()})

    stats = measure_ms_per_packet(lambda: torch.randn(64, 64) @ torch.randn(64, 64),
                                  n_packets=10)
    print("ms/packet (matmul) =", {k: round(v, 3) for k, v in stats.items()})
    print("peak VRAM          =", peak_vram_mb(), "(None expected without CUDA)")
    print("eval.py self-check OK")
