r"""
flops.py -- a simple, hardware-INDEPENDENT FLOP proxy for the LIF-gated update (A4).

Wall-clock ms depends on the CPU/GPU; to make the efficiency claim portable we also report an
analytic FLOP count of the per-window OPTIMISATION work, derived from the exact operations of
the additive 2D renderer (splat2d). It is a *proxy* -- transcendental ops (exp, sigmoid) are
counted as 1 FLOP -- so read the RATIO across settings, not the absolute magnitude.

One forward render costs RENDER_FLOPS elementwise ops per (pixel, Gaussian):
    diff (2 sub) + sq (2 mul, 1 add) + G (1 div, 1 mul, 1 exp) + alpha*G (1 mul)
    + weighted sum (1 mul, 1 add)  ~= 11
A differentiated render costs ~FWD_BWD x its forward cost (backward ~ 2x forward).

Per window the bounded optimiser does exactly what run_budget.py times:
  - inactive cache : 2 no-grad renders of (N - A) Gaussians, ONCE  -> 2*(N-A)*HW, forward only
  - active steps   : S steps, each 2 differentiated renders of A    -> S*2*A*HW, fwd+bwd (3x)
Hence
    flops(window) = RENDER_FLOPS * HW * [ 2*(N-A)   +   FWD_BWD * S * 2 * A ]
                                         \_________/     \__________________/
                                          O(N) cache      bounded by B  (A <= B)
The first term is the honest O(N) overhead -- the reason rasteriser-level culling only pays at
large N (Table `tab:cull`); the second is the part Prop.4 bounds by the budget B.
"""
from __future__ import annotations

RENDER_FLOPS = 11      # elementwise ops per (pixel, Gaussian) in one forward render
FWD_BWD = 3            # a differentiated render ~ 3x its forward cost


def budget_window_mflops(n_total: int, n_active: float, hw: int, steps: int) -> float:
    """MFLOP proxy for one window's bounded optimisation (inactive cache + active steps)."""
    cache = 2.0 * (n_total - n_active) * hw               # forward-only, once per window
    active = FWD_BWD * steps * 2.0 * n_active * hw        # fwd+bwd, per step, 2 renders each
    return RENDER_FLOPS * (cache + active) / 1e6


def render_mflops(n_render: float, hw: int, differentiated: bool = False) -> float:
    """MFLOP proxy for a single render of n_render Gaussians over hw pixels."""
    return RENDER_FLOPS * n_render * hw * (FWD_BWD if differentiated else 1) / 1e6
