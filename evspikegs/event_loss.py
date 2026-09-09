"""
event_loss.py
-------------
The differentiable event photometric loss (this REPLACES any hand-crafted update rule).

We supervise the *difference* of rendered log-intensities against the accumulated events:

    dLhat(u) = Lhat(u, t_curr) - Lhat(u, t_prev)          # predicted log-intensity change
    L_event  = sum_u  rho( dLhat(u) - dE(u) )             # rho = Huber

Everything is differentiable, so the scene is optimized by gradient descent, exactly as in
standard 3DGS but with event supervision instead of RGB supervision. No absolute intensity
is required anywhere in this loss (see event_data.py).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def event_photometric_loss(dLhat: torch.Tensor, dE: torch.Tensor, delta: float = 0.1):
    """Huber loss between predicted and observed log-intensity change."""
    return F.huber_loss(dLhat, dE, delta=delta, reduction="mean")


def anchor_loss(img_pred: torch.Tensor, img_ref: torch.Tensor):
    """
    Optional single-keyframe absolute-brightness anchor (the role of the sparse RGB frames
    in the real pipeline). It removes the global radiance-scale ambiguity inherent to
    event-only supervision. Used once, at the first window.
    """
    return F.l1_loss(img_pred, img_ref)
