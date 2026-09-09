"""
event_data.py
-------------
Synthetic event generation that faithfully follows the event-camera model:

    an event fires when the change in LOG-intensity crosses a contrast threshold C:
        L(t) - L(t^-) = p * C,   L = log I,   p in {-1, +1}

Key point (this is what fixes the classic mistake of using absolute intensity):
events encode the *temporal difference* of log-intensity, never the absolute intensity.

For the toy 2D demo we obtain L(t) by rendering a fixed ground-truth scene at a sequence
of camera poses (2D translations) and taking log. Between consecutive windows we quantize
the log-intensity change into a signed integer number of contrast steps, then multiply by C
to get the accumulated event map  dE(u)  used to supervise the model.
"""

from __future__ import annotations
import torch
from .splat2d import render, log_intensity


def accumulate_events(L_prev: torch.Tensor, L_curr: torch.Tensor, C: float):
    """
    Quantize the log-intensity change into events and return the accumulated map dE.

        dE(u) = C * round( (L_curr(u) - L_prev(u)) / C )

    This mimics integrating the polarity*C of all events at pixel u within the window.
    """
    dL = L_curr - L_prev
    n_steps = torch.round(dL / C)          # signed integer number of threshold crossings
    dE = n_steps * C
    return dE


def make_circular_poses(n_windows: int, radius: float = 2.0):
    """A small circular camera trajectory (2D translations), one pose per window."""
    t = torch.linspace(0, 2 * torch.pi, n_windows)
    return torch.stack([radius * torch.cos(t), radius * torch.sin(t)], dim=1)  # (T,2)


def gt_log_at_pose(gt_scene, grid, pose):
    """Render the ground-truth scene as seen after a 2D camera translation `pose`."""
    img = render(gt_scene, grid + pose)     # shifting the query grid == translating the camera
    return log_intensity(img), img
