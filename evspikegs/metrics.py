"""metrics.py -- image metrics (PSNR, MAE, SSIM).
LPIPS needs a pretrained net (`pip install lpips`) and is used only in the GPU eval."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return (10.0 * torch.log10((max_val ** 2) / mse)).item()


def mae(pred, target):
    return torch.mean((pred - target).abs()).item()


def _gaussian_kernel(window=11, sigma=1.5):
    x = torch.arange(window, dtype=torch.float32) - (window - 1) / 2.0
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]     # (1,1,win,win)


def ssim(pred, target, max_val=1.0, window=11, sigma=1.5):
    """Standard single-scale SSIM for single-channel images of shape (H, W)."""
    p = pred.reshape(1, 1, *pred.shape[-2:]).float()
    t = target.reshape(1, 1, *target.shape[-2:]).float()
    k = _gaussian_kernel(window, sigma).to(p.device)
    pad = window // 2
    mu_p = F.conv2d(p, k, padding=pad)
    mu_t = F.conv2d(t, k, padding=pad)
    mu_pp, mu_tt, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t
    var_p = F.conv2d(p * p, k, padding=pad) - mu_pp
    var_t = F.conv2d(t * t, k, padding=pad) - mu_tt
    cov = F.conv2d(p * t, k, padding=pad) - mu_pt
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    s = ((2 * mu_pt + c1) * (2 * cov + c2)) / ((mu_pp + mu_tt + c1) * (var_p + var_t + c2))
    return s.mean().item()
