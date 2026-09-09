"""
esim_sim.py -- ESIM-style event simulator (per-event timestamps, reference-level model).

Implements the standard event-camera simulation used by ESIM (Rebecq et al., CoRL'18):
each pixel keeps a reference log-intensity L_ref; as the log-intensity signal L(t) evolves
(sampled at frame times, linearly interpolated in between), an event (x, y, t, p) is emitted
every time |L(t) - L_ref| reaches the contrast threshold C, at the linearly-interpolated
crossing time, and L_ref moves by p*C. Multiple events per frame interval are emitted when
the change exceeds several thresholds.

This upgrades the window-quantised generator in event_data.py to a genuine asynchronous
event STREAM, so the full packet pipeline (datasets.EventPacket consumers) can be exercised
under a controlled, ground-truth-available simulation — the role ESIM plays in the
experimental protocol, without the ROS dependency (exact same sensor model).
"""
from __future__ import annotations
from typing import Tuple

import torch


@torch.no_grad()
def simulate_events(log_frames: torch.Tensor, frame_ts: torch.Tensor,
                    C: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
    """
    Emit an asynchronous event stream from a sequence of log-intensity frames.

    log_frames: (T, H, W) log-intensity at sample times; frame_ts: (T,) increasing times.
    Returns (x, y, t, p) int32/float tensors sorted by t. Linear interpolation between
    frames gives each event its sub-frame timestamp, exactly as in ESIM.
    """
    T, H, W = log_frames.shape
    Lref = log_frames[0].clone()                       # per-pixel reference level
    xs, ys, ts, ps = [], [], [], []
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    for i in range(1, T):
        L0, L1 = log_frames[i - 1], log_frames[i]
        t0, t1 = float(frame_ts[i - 1]), float(frame_ts[i])
        dL = L1 - Lref
        n_cross = torch.floor(dL.abs() / C).long()     # events per pixel this interval
        max_n = int(n_cross.max())
        for k in range(1, max_n + 1):
            m = n_cross >= k
            if not m.any():
                break
            pol = torch.sign(dL[m])
            # crossing level and its linear-interpolated time within [t0, t1]
            level = Lref[m] + pol * k * C
            denom = (L1[m] - L0[m])
            frac = ((level - L0[m]) / torch.where(denom.abs() < 1e-12,
                                                  torch.full_like(denom, 1e-12), denom))
            frac = frac.clamp(0.0, 1.0)
            xs.append(xx[m]); ys.append(yy[m])
            ts.append(torch.as_tensor(t0) + frac * (t1 - t0))
            ps.append(pol)
        # move the reference by the emitted quanta
        Lref = Lref + torch.sign(dL) * n_cross.to(Lref.dtype) * C

    if not xs:
        z = torch.zeros(0)
        return z.int(), z.int(), z, z
    x = torch.cat(xs).int(); y = torch.cat(ys).int()
    t = torch.cat(ts).float(); p = torch.cat(ps).float()
    order = torch.argsort(t)
    return x[order], y[order], t[order], p[order]


@torch.no_grad()
def events_to_window_maps(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                          p: torch.Tensor, H: int, W: int, n_windows: int,
                          C: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cut the stream into equal windows; return (dE maps (n,H*W), window edge times)."""
    edges = torch.linspace(float(t.min()), float(t.max()) + 1e-9, n_windows + 1)
    maps = torch.zeros(n_windows, H * W)
    idx = (y.long() * W + x.long())
    win = torch.clamp(torch.searchsorted(edges, t, right=True) - 1, 0, n_windows - 1)
    flat = win * (H * W) + idx
    maps.view(-1).scatter_add_(0, flat, p * C)
    return maps, edges
