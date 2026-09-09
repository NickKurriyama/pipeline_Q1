"""
render3d.py  --  real 3D geometry for EvSpike-GS.

Contains:
  * PinholeCamera         : world->camera->image projection with intrinsics/extrinsics
  * Gaussians3D           : learnable 3D Gaussian parameters (mean, quat, log-scale, opacity, brightness)
  * render_torch(...)     : a MINIMAL differentiable 3D Gaussian rasterizer in pure PyTorch that
                            runs on CPU (real perspective projection + depth-sorted alpha
                            compositing). Isotropic splats to keep it small and robust; this is
                            enough to reconstruct a toy 3D scene from events end-to-end.
  * render_gsplat(...)    : the real backend for GPU + real data (wraps gsplat.rasterization).

The two renderers share the SAME interface:
    image (H,W), weights (H*W, N), depth (N,)
so trainer3d.py is backend-agnostic. Use 'torch' for the CPU smoke test (train_synthetic.py),
'gsplat' for EventNeRF/DSEC on a GPU.
"""
from __future__ import annotations
from typing import Optional
import torch


# --------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------
class PinholeCamera:
    def __init__(self, H, W, fx, fy, cx, cy, R, t):
        """R (3,3), t (3,): world->camera, i.e. x_cam = R @ x_world + t."""
        self.H, self.W = H, W
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.R, self.t = R, t

    @staticmethod
    def look_at(eye, target, up, H, W, fov_deg=60.0):
        eye = torch.as_tensor(eye, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        up = torch.as_tensor(up, dtype=torch.float32)
        f = (target - eye); f = f / f.norm()
        r = torch.linalg.cross(f, up); r = r / r.norm()
        u = torch.linalg.cross(r, f)
        # camera looks along +z (OpenCV-style): rows are right, down, forward
        Rcw = torch.stack([r, -u, f], dim=0)      # world->camera rotation
        t = -Rcw @ eye
        fx = fy = 0.5 * W / torch.tan(torch.tensor(fov_deg * torch.pi / 360.0))
        return PinholeCamera(H, W, float(fx), float(fy), W / 2.0, H / 2.0, Rcw, t)

    def project(self, X_world):
        """X_world (N,3) -> (uv (N,2), depth (N,)). Follows X_world's device."""
        R = self.R.to(X_world.device)
        t = self.t.to(X_world.device)
        X_cam = X_world @ R.T + t                                # (N,3)
        z = X_cam[:, 2].clamp_min(1e-4)
        u = self.fx * X_cam[:, 0] / z + self.cx
        v = self.fy * X_cam[:, 1] / z + self.cy
        return torch.stack([u, v], dim=1), z


# --------------------------------------------------------------------------------------
# 3D Gaussians
# --------------------------------------------------------------------------------------
class Gaussians3D:
    def __init__(self, means, log_scales, quats, opacity_logit, brightness, learnable=True):
        self.means = means.clone().requires_grad_(learnable)          # (N,3)
        self.log_scales = log_scales.clone().requires_grad_(learnable)  # (N,3)
        self.quats = quats.clone().requires_grad_(learnable)          # (N,4) wxyz
        self.opacity_logit = opacity_logit.clone().requires_grad_(learnable)  # (N,)
        self.brightness = brightness.clone().requires_grad_(learnable)  # (N,) grayscale emission

    @property
    def params(self):
        return [self.means, self.log_scales, self.quats, self.opacity_logit, self.brightness]

    @property
    def n(self):
        return self.means.shape[0]

    def iso_scale(self):
        """Isotropic world-space radius used by the minimal torch rasterizer."""
        return torch.exp(self.log_scales).mean(dim=1)                 # (N,)

    def to(self, device) -> "Gaussians3D":
        """Move all learnable parameters to `device` in place (keeps requires_grad)."""
        for p in self.params:
            p.data = p.data.to(device)
            if p.grad is not None:
                p.grad = p.grad.to(device)
        return self

    @staticmethod
    def random(n, center=(0, 0, 0), spread=1.0, scale=0.08, seed=0, brightness=0.7):
        g = torch.Generator().manual_seed(seed)
        c = torch.tensor(center, dtype=torch.float32)
        means = c + spread * (torch.rand(n, 3, generator=g) - 0.5)
        log_scales = torch.log(torch.full((n, 3), float(scale)))
        quats = torch.zeros(n, 4); quats[:, 0] = 1.0                  # identity rotation
        opacity_logit = torch.full((n,), -2.0) + 0.1 * torch.randn(n, generator=g)
        b = torch.full((n,), float(brightness)) + 0.05 * torch.randn(n, generator=g)
        return Gaussians3D(means, log_scales, quats, opacity_logit, b)

    @torch.no_grad()
    def clamp_(self, max_scale=0.5):
        """Box constraints after each step — prevents the brightness/opacity blow-up
        degenerate optimum under very sparse (gated) updates."""
        self.brightness.data.clamp_(0.0, 1.2)
        self.opacity_logit.data.clamp_(-6.0, 4.0)
        self.log_scales.data.clamp_(float(torch.log(torch.tensor(1e-3))),
                                    float(torch.log(torch.tensor(max_scale))))


# --------------------------------------------------------------------------------------
# Minimal pure-PyTorch 3D rasterizer (CPU-friendly)
# --------------------------------------------------------------------------------------
def _pixel_grid(H, W, device=None):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                            torch.arange(W, dtype=torch.float32, device=device),
                            indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)      # (HW,2)


def render_torch(g: Gaussians3D, cam: PinholeCamera, eps=1e-3, bg: float = 0.0):
    """
    Real perspective projection + depth-sorted alpha compositing (isotropic splats).
    Returns: image (H,W) in [0,1], weights (HW,N) blending weights T_i*alpha_i, depth (N,).
    Device-agnostic: runs on CPU or CUDA according to where the Gaussians live.
    `bg` is the background brightness composited behind all Gaussians (real captures
    rarely have black backgrounds — mismatching it poisons the anchor loss).
    """
    grid = _pixel_grid(cam.H, cam.W, device=g.means.device)          # (HW,2)
    uv, z = cam.project(g.means)                                     # (N,2),(N,)
    radius = (g.iso_scale() * cam.fx / z).clamp(0.5, max(cam.H, cam.W))  # (N,) px
    # per-pixel per-gaussian Gaussian value
    d2 = ((grid[:, None, :] - uv[None, :, :]) ** 2).sum(-1)          # (HW,N)
    G = torch.exp(-0.5 * d2 / (radius[None, :] ** 2))                # (HW,N)
    alpha = torch.sigmoid(g.opacity_logit)[None, :] * G             # (HW,N)

    # depth-sorted front-to-back alpha compositing (order detached, values differentiable)
    order = torch.argsort(z)                                         # near->far
    A = alpha[:, order]                                             # (HW,N) sorted
    col = g.brightness[order][None, :]                              # (1,N)
    one_minus = (1.0 - A).clamp(1e-6, 1.0)
    T = torch.cumprod(one_minus, dim=1)                            # inclusive
    T_before = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1)  # transmittance before i
    w_sorted = T_before * A                                        # (HW,N) blending weights
    image = ((w_sorted * col).sum(dim=1) + T[:, -1] * bg).clamp(0.0, 1.0)  # (HW,)

    # unsort weights back to original gaussian indexing (for LIF stimulus)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    weights = w_sorted[:, inv]                                     # (HW,N)
    return image.reshape(cam.H, cam.W), weights, z


def render_gsplat(g: Gaussians3D, cam: PinholeCamera, bg: float = 0.0):
    """
    Real backend for GPU + real data. Requires `pip install gsplat` and a CUDA device.
    Returns the same (image, weights, depth) interface, except `weights` is None:
    gsplat does not expose dense per-pixel per-Gaussian blending weights, so the LIF
    stimulus is computed by `stimulus_from_projection` instead (trainer3d handles this).

    NOTE: written against gsplat >= 1.0 (`gsplat.rasterization`); untested without CUDA —
    verify shapes/kwargs against your installed version before a real run.
    """
    from gsplat import rasterization

    device = g.means.device
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = cam.R.to(device)
    viewmat[:3, 3] = cam.t.to(device)
    K = torch.tensor([[cam.fx, 0.0, cam.cx],
                      [0.0, cam.fy, cam.cy],
                      [0.0, 0.0, 1.0]], device=device)

    colors = g.brightness[:, None].expand(-1, 3).contiguous()   # grayscale as RGB
    render, _alpha, _info = rasterization(
        means=g.means,
        quats=g.quats / g.quats.norm(dim=1, keepdim=True).clamp_min(1e-8),
        scales=torch.exp(g.log_scales),
        opacities=torch.sigmoid(g.opacity_logit),
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=cam.W,
        height=cam.H,
        render_mode="RGB",
        # NOTE: with packed=True (default), gsplat 1.5.3 wants backgrounds of shape (D,)
        # — probed empirically; the docstring's [..., C, D] only holds for packed=False
        backgrounds=torch.full((3,), float(bg), device=device),
    )
    image = render[0, ..., :3].mean(dim=-1).clamp(0.0, 1.0)     # (H,W) grayscale
    _, z = cam.project(g.means)
    return image, None, z


def _gsplat_call(means, quats, scales, opacities, colors, cam, bg_scalar, mode="RGB"):
    """Thin wrapper around gsplat.rasterization for an explicit primitive subset."""
    from gsplat import rasterization
    device = means.device
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = cam.R.to(device)
    viewmat[:3, 3] = cam.t.to(device)
    K = torch.tensor([[cam.fx, 0.0, cam.cx],
                      [0.0, cam.fy, cam.cy],
                      [0.0, 0.0, 1.0]], device=device)
    return rasterization(
        means=means,
        quats=quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-8),
        scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmat[None], Ks=K[None], width=cam.W, height=cam.H,
        render_mode=mode,
        backgrounds=torch.full((3,), float(bg_scalar), device=device),
    )


def render_gsplat_subset(g: Gaussians3D, cam: PinholeCamera, idx: torch.Tensor,
                         bg_image: Optional[torch.Tensor] = None, bg: float = 0.0):
    """
    Rasterise ONLY the Gaussians in `idx` (the LIF active set) and composite them over a
    cached image of everything else. This is where the gate's compute saving is realised:
    non-firing Gaussians never enter the forward or the backward pass, so both scale with
    |idx| rather than N. Indexing keeps the graph connected to the original leaf tensors,
    so gradients accumulate only into the selected rows.

    APPROXIMATION (stated, not hidden): the cached inactive image is composited as a
    single background layer, so active Gaussians are always blended in front of it.
    Depth interleaving between active and inactive primitives within a window is ignored;
    the error is confined to pixels where a firing Gaussian is occluded by a frozen one,
    and it is corrected at the next window, when the cache is rebuilt.
    """
    if idx.numel() == 0:
        base = bg_image if bg_image is not None else torch.full(
            (cam.H, cam.W), float(bg), device=g.means.device)
        return base.clamp(0.0, 1.0)
    colors = g.brightness[idx][:, None].expand(-1, 3).contiguous()
    render, alpha, _ = _gsplat_call(g.means[idx], g.quats[idx],
                                    torch.exp(g.log_scales[idx]),
                                    torch.sigmoid(g.opacity_logit[idx]),
                                    colors, cam, 0.0)
    img_active = render[0, ..., :3].mean(dim=-1)
    a = alpha[0, ..., 0]
    base = bg_image if bg_image is not None else float(bg)
    return (img_active + (1.0 - a) * base).clamp(0.0, 1.0)


@torch.no_grad()
def render_gsplat_inactive_cache(g: Gaussians3D, cam: PinholeCamera,
                                 active_mask: torch.Tensor, bg: float = 0.0):
    """Detached image of the frozen (non-firing) Gaussians — rendered once per window."""
    idx = (~active_mask).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.full((cam.H, cam.W), float(bg), device=g.means.device)
    colors = g.brightness[idx][:, None].expand(-1, 3).contiguous()
    render, _alpha, _ = _gsplat_call(g.means[idx], g.quats[idx],
                                     torch.exp(g.log_scales[idx]),
                                     torch.sigmoid(g.opacity_logit[idx]),
                                     colors, cam, bg)
    return render[0, ..., :3].mean(dim=-1).clamp(0.0, 1.0).detach()


@torch.no_grad()
def render_depth_gsplat(g: Gaussians3D, cam: PinholeCamera) -> torch.Tensor:
    """Expected-depth map (H,W) from gsplat (render_mode='ED'); 0 where nothing renders.
    Used for depth-vs-GT evaluation (MVSEC/DSEC tier)."""
    from gsplat import rasterization
    device = g.means.device
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = cam.R.to(device)
    viewmat[:3, 3] = cam.t.to(device)
    K = torch.tensor([[cam.fx, 0.0, cam.cx],
                      [0.0, cam.fy, cam.cy],
                      [0.0, 0.0, 1.0]], device=device)
    render, _alpha, _ = rasterization(
        means=g.means,
        quats=g.quats / g.quats.norm(dim=1, keepdim=True).clamp_min(1e-8),
        scales=torch.exp(g.log_scales),
        opacities=torch.sigmoid(g.opacity_logit),
        colors=g.brightness[:, None].expand(-1, 3).contiguous(),
        viewmats=viewmat[None], Ks=K[None], width=cam.W, height=cam.H,
        render_mode="ED",
    )
    return render[0, ..., 0]


@torch.no_grad()
def stimulus_from_projection(g: Gaussians3D, cam: PinholeCamera, residual_flat: torch.Tensor):
    """
    Cheap LIF stimulus when the backend does not expose per-Gaussian blending weights:
    bilinearly sample the |event residual| image at each Gaussian's projected center and
    weight by its opacity and screen-space footprint area. This approximates
    s_k = sum_u w_{k,u} |dE - dLhat|(u) with a single point sample per Gaussian (O(N)).
    """
    res = residual_flat.reshape(1, 1, cam.H, cam.W)
    uv, z = cam.project(g.means)
    # normalized grid coords in [-1, 1]
    gx = (uv[:, 0] / (cam.W - 1)) * 2 - 1
    gy = (uv[:, 1] / (cam.H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=1).reshape(1, -1, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        res, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(-1)                                # (N,)
    radius_px = (g.iso_scale() * cam.fx / z.clamp_min(1e-4)).clamp(0.5, max(cam.H, cam.W))
    footprint = radius_px ** 2                                   # ~ area in pixels
    return torch.sigmoid(g.opacity_logit) * footprint * sampled
