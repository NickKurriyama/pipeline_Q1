"""
splat2d.py
----------
A minimal *differentiable* 2D Gaussian splatting renderer, pure PyTorch, CPU-friendly.

This is a TOY renderer whose only purpose is to let the novel EvSpike-GS mechanisms
(event photometric loss + LIF-gated sparse updates) be demonstrated end-to-end on a
laptop, without a CUDA rasterizer. The real 3D pipeline (see ../pipeline3d) should use
an established differentiable 3D rasterizer such as `gsplat` or `diff-gaussian-rasterization`.

Scene = a set of N 2D isotropic Gaussians, each with:
  - mean      mu   in image coordinates            (N,2)
  - log-scale ls   (isotropic std = exp(ls))        (N,)
  - brightness b   (grayscale emission, >=0)        (N,)
  - opacity   o    (logit; alpha = sigmoid(o))      (N,)

Rendering uses simplified additive (emission) splatting; it is intentionally NOT the full
EWA ordered alpha-compositing of real 3DGS. It is sufficient to exhibit the mechanism.
"""

from __future__ import annotations
import torch


class GaussianScene2D:
    """Container of learnable 2D Gaussian parameters."""

    def __init__(self, mu, ls, b, o, learnable=True):
        self.mu = mu.clone().requires_grad_(learnable)
        self.ls = ls.clone().requires_grad_(learnable)
        self.b = b.clone().requires_grad_(learnable)
        self.o = o.clone().requires_grad_(learnable)

    @property
    def params(self):
        return [self.mu, self.ls, self.b, self.o]

    @property
    def n(self):
        return self.mu.shape[0]

    @staticmethod
    def random(n, H, W, seed=0, brightness=0.7):
        g = torch.Generator().manual_seed(seed)
        mu = torch.rand(n, 2, generator=g) * torch.tensor([W - 1.0, H - 1.0])
        ls = torch.log(torch.full((n,), max(H, W) / 20.0)) + 0.2 * torch.randn(n, generator=g)
        b = torch.full((n,), float(brightness)) + 0.05 * torch.randn(n, generator=g)
        # start nearly transparent so the image is dim; events + anchor build it up
        o = torch.full((n,), -2.5) + 0.1 * torch.randn(n, generator=g)
        return GaussianScene2D(mu, ls, b, o)

    @torch.no_grad()
    def clamp_(self, H, W):
        """
        Project parameters back into their physically valid box after each optimizer step.
        Without this, very sparse updates can push a few repeatedly-firing Gaussians into a
        DEGENERATE optimum: brightness explodes, the image saturates at the clamp, the event
        residual (and its gradient) collapses to zero, and optimization deadlocks.
        """
        self.b.data.clamp_(0.0, 1.2)                     # bounded emission (GT <= 1)
        self.o.data.clamp_(-6.0, 4.0)                    # keep sigmoid out of dead zones
        self.ls.data.clamp_(float(torch.log(torch.tensor(0.5))),
                            float(torch.log(torch.tensor(max(H, W) / 4.0))))
        self.mu.data[:, 0].clamp_(-0.25 * W, 1.25 * W)   # stay near the canvas
        self.mu.data[:, 1].clamp_(-0.25 * H, 1.25 * H)


def base_grid(H, W):
    """Return pixel coordinate grid of shape (H*W, 2) as (x, y)."""
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32),
                            indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (HW, 2)


def gaussian_weights(scene: GaussianScene2D, grid: torch.Tensor):
    """
    Per-pixel, per-Gaussian weight w_{p,i} = alpha_i * exp(-||p - mu_i||^2 / (2 sigma_i^2)).
    Returns w of shape (HW, N).
    """
    diff = grid[:, None, :] - scene.mu[None, :, :]        # (HW, N, 2)
    sq = (diff ** 2).sum(-1)                              # (HW, N)
    sigma2 = torch.exp(2.0 * scene.ls)[None, :]           # (1, N)
    G = torch.exp(-0.5 * sq / sigma2)                     # (HW, N)
    alpha = torch.sigmoid(scene.o)[None, :]               # (1, N)
    return alpha * G                                       # (HW, N)


def render(scene: GaussianScene2D, grid: torch.Tensor):
    """
    Additive emission rendering. Returns image intensity of shape (HW,), clamped to [0, 1].
    """
    w = gaussian_weights(scene, grid)                     # (HW, N)
    img = (w * scene.b[None, :]).sum(dim=1)               # (HW,)
    return img.clamp(0.0, 1.0)


def log_intensity(img, eps=1e-3):
    return torch.log(img + eps)


# ------------------------------------------------------------------------------------------
# Active-set (partial) rendering -- realizes the REAL compute saving of the LIF gate.
#
# Because this toy renderer is additive, the image splits EXACTLY into
#     img = clamp( inactive_contribution + active_contribution )
# where the inactive part is constant within a window (its Gaussians are frozen), so it is
# rendered ONCE per window under no_grad and cached; each inner optimization step then only
# renders + differentiates the |active| << N Gaussians. This is the toy analogue of culling
# non-firing Gaussians from a real rasterizer's forward/backward pass.
# ------------------------------------------------------------------------------------------
def render_subset_raw(scene: GaussianScene2D, grid: torch.Tensor, idx: torch.Tensor):
    """Unclamped additive contribution of the Gaussians in `idx`. Shape (HW,)."""
    if idx.numel() == 0:
        return torch.zeros(grid.shape[0])
    diff = grid[:, None, :] - scene.mu[idx][None, :, :]   # (HW, K, 2)
    sq = (diff ** 2).sum(-1)
    sigma2 = torch.exp(2.0 * scene.ls[idx])[None, :]
    G = torch.exp(-0.5 * sq / sigma2)
    w = torch.sigmoid(scene.o[idx])[None, :] * G
    return (w * scene.b[idx][None, :]).sum(dim=1)


@torch.no_grad()
def render_inactive_cache(scene: GaussianScene2D, grid: torch.Tensor, active_mask: torch.Tensor):
    """Detached contribution of the non-active Gaussians (computed once per window)."""
    idx = (~active_mask).nonzero(as_tuple=True)[0]
    return render_subset_raw(scene, grid, idx).detach()


def render_active(scene: GaussianScene2D, grid: torch.Tensor, active_mask: torch.Tensor,
                  inactive_cache: torch.Tensor):
    """Full image = cached inactive part + differentiable active part (exact split)."""
    idx = active_mask.nonzero(as_tuple=True)[0]
    return (inactive_cache + render_subset_raw(scene, grid, idx)).clamp(0.0, 1.0)
