"""Tests for the pure-PyTorch 3D rasterizer: image range, depth ordering, gradient flow,
and the (image, weights, depth) interface contract the trainer relies on."""
from __future__ import annotations
import torch

from render3d import PinholeCamera, Gaussians3D, render_torch

H = W = 24


def _cam(radius: float = 2.0) -> PinholeCamera:
    return PinholeCamera.look_at((0, 0, radius), (0, 0, 0), (0, 1, 0), H, W, fov_deg=60.0)


def _one_gaussian(z: float = 0.0, brightness: float = 0.9) -> Gaussians3D:
    means = torch.tensor([[0.0, 0.0, z]])
    log_scales = torch.log(torch.full((1, 3), 0.15))
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    opacity = torch.tensor([2.0])
    b = torch.tensor([brightness])
    return Gaussians3D(means, log_scales, quats, opacity, b)


def test_image_in_unit_range_and_correct_shape():
    img, w, z = render_torch(_one_gaussian(), _cam())
    assert img.shape == (H, W)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert w.shape == (H * W, 1)
    assert z.shape == (1,)


def test_depth_matches_camera_distance():
    cam = _cam(radius=2.0)
    _, _, z = render_torch(_one_gaussian(z=0.0), cam)
    assert abs(z.item() - 2.0) < 1e-4, "Gaussian at origin, camera at r=2 -> depth 2"
    _, _, z_near = render_torch(_one_gaussian(z=0.5), cam)   # closer to the camera
    assert z_near.item() < z.item()


def test_center_pixel_brighter_than_corner():
    img, _, _ = render_torch(_one_gaussian(), _cam())
    assert img[H // 2, W // 2] > img[0, 0]


def test_occlusion_front_gaussian_dominates():
    """Two Gaussians on the optical axis: the nearer one must dominate the center pixel."""
    means = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])   # first is nearer to cam at +z
    log_scales = torch.log(torch.full((2, 3), 0.15))
    quats = torch.zeros(2, 4); quats[:, 0] = 1.0
    opacity = torch.full((2,), 4.0)                              # nearly opaque
    b = torch.tensor([1.0, 0.0])                                 # bright front, dark back
    g = Gaussians3D(means, log_scales, quats, opacity, b)
    img, w, _ = render_torch(g, _cam())
    center = H // 2 * W + W // 2
    assert img[H // 2, W // 2] > 0.5, "front bright Gaussian must win the center pixel"
    assert w[center, 0] > w[center, 1], "front Gaussian gets the larger blending weight"


def test_gradients_flow_to_means_and_brightness():
    g = _one_gaussian()
    img, _, _ = render_torch(g, _cam())
    (img ** 2).mean().backward()
    assert g.means.grad is not None and g.means.grad.abs().sum() > 0
    assert g.brightness.grad is not None and g.brightness.grad.abs().sum() > 0


def test_weights_unsorted_back_to_input_indexing():
    """weights[:, k] must correspond to input Gaussian k regardless of depth order."""
    means = torch.tensor([[0.4, 0.0, -0.5], [-0.4, 0.0, 0.5]])  # index 1 is nearer
    log_scales = torch.log(torch.full((2, 3), 0.12))
    quats = torch.zeros(2, 4); quats[:, 0] = 1.0
    g = Gaussians3D(means, log_scales, quats, torch.full((2,), 2.0), torch.full((2,), 0.8))
    _, w, z = render_torch(g, _cam())
    assert z[1] < z[0]
    # each Gaussian's weight mass must sit on its own side of the image
    wmap0 = w[:, 0].reshape(H, W); wmap1 = w[:, 1].reshape(H, W)
    left, right = slice(0, W // 2), slice(W // 2, W)
    # OpenCV convention: camera x-axis aligns with world +x here, so +x -> image RIGHT
    assert wmap0[:, right].sum() > wmap0[:, left].sum()
    assert wmap1[:, left].sum() > wmap1[:, right].sum()


def test_clamp_keeps_parameters_in_box():
    g = _one_gaussian()
    with torch.no_grad():
        g.brightness += 100.0
        g.opacity_logit += 100.0
    g.clamp_()
    assert g.brightness.max() <= 1.2 and g.opacity_logit.max() <= 4.0
