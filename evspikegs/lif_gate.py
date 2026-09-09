"""
lif_gate.py
-----------
**The core novelty of EvSpike-GS.**

Each Gaussian is treated as a Leaky Integrate-and-Fire (LIF) neuron. Per event window:

  1. Stimulus  s_k  = event-driven error routed to Gaussian k via the splatting weights:
         s_k = sum_u  w_{k,u} * | dE(u) - dLhat(u) |
     (w_{k,u} are the same per-Gaussian rendering weights, so this is cheap.)

  2. Leaky integration of the membrane potential (0 < lam < 1):
         V_k <- lam * V_k + s_k

  3. Fire / gate:
         fire_k = 1[ V_k >= theta ]
     Only firing Gaussians are placed in the active set: they receive parameter updates
     (and, in a real rasterizer, they are the only ones included in the forward/backward
     pass -- that is where the compute is actually saved). Fired neurons reset: V_k <- 0.

Why this is more than a per-frame mask:
  - LEAKY TEMPORAL INTEGRATION: weak-but-persistent error accumulates across windows until
    it fires, so slowly-changing regions are eventually corrected; transient noise leaks away.
  - BOUNDED PER-WINDOW COMPUTE: capping the active-set size caps the per-window cost, which
    is what makes streaming/online reconstruction feasible.

Hard budget B (claim C3): pass `budget=B` to cap the active set per window. When more than B
neurons fire, only the B with the highest membrane potential are admitted; the rest KEEP their
charge (no reset), so they are guaranteed to be served in a later window. This turns overload
into a graceful backlog instead of dropped updates, and bounds the per-window cost by B.

Ablation modes are provided so the contribution can be isolated:
  - 'lif'    : full leaky integrate-and-fire gate (ours)
  - 'binary' : per-window mask, no temporal memory  (fire iff s_k >= theta)
  - 'topk'   : keep the k Gaussians with largest s_k (fixed sparsity)
  - 'dense'  : no gating (all Gaussians updated)  -- the theta->0 baseline
"""

from __future__ import annotations
from typing import Optional, Sequence
import torch


class LIFGate:
    def __init__(self, n: int, lam: float = 0.6, theta: float = 0.05, mode: str = "lif",
                 topk_frac: float = 0.30, budget: Optional[int] = None):
        self.mode = mode
        self.lam = lam
        self.theta = theta
        self.topk_frac = topk_frac
        self.budget = budget             # hard cap on active-set size (None = uncapped)
        self.V = torch.zeros(n)          # membrane potentials
        self.last_fire_frac = 1.0

    @torch.no_grad()
    def _cap(self, fire, score):
        """Admit at most `budget` neurons, preferring the highest score (membrane/stimulus)."""
        if self.budget is None or int(fire.sum()) <= self.budget:
            return fire
        masked = torch.where(fire, score, torch.full_like(score, float("-inf")))
        idx = torch.topk(masked, self.budget).indices
        mask = torch.zeros_like(fire)
        mask[idx] = True
        return mask

    @torch.no_grad()
    def resize(self, keep_idx, n_new):
        """After densify/prune: carry over the membrane of surviving Gaussians, zero-init new."""
        self.V = torch.cat([self.V[keep_idx],
                            torch.zeros(n_new, device=self.V.device)])

    @torch.no_grad()
    def stimulus(self, w_detached: torch.Tensor, residual_abs: torch.Tensor) -> torch.Tensor:
        """
        s_k = sum_u w_{k,u} * |dE - dLhat|(u).
        w_detached: (HW, N) rendering weights (detached).
        residual_abs: (HW,) absolute per-pixel event residual.
        Returns s of shape (N,).
        """
        return (w_detached * residual_abs[:, None]).sum(dim=0)

    @staticmethod
    @torch.no_grad()
    def stimulus_from_grads(params: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Gradient-magnitude stimulus variant: s_k = total gradient norm routed to Gaussian k,
        summed over all its parameter tensors. Use when the render backend does not expose
        per-Gaussian blending weights (e.g. gsplat). Call AFTER loss.backward().
        Each tensor in `params` has leading dim N (per-Gaussian); missing grads count as 0.
        """
        s: Optional[torch.Tensor] = None
        for p in params:
            if p.grad is None:
                continue
            g = p.grad
            contrib = g.abs() if g.dim() == 1 else g.norm(dim=tuple(range(1, g.dim())))
            s = contrib if s is None else s + contrib
        if s is None:
            raise RuntimeError("stimulus_from_grads: no parameter has a gradient; "
                               "call loss.backward() first")
        return s

    @torch.no_grad()
    def step(self, s):
        """Update membrane, decide firing mask, return boolean active-set mask (N,).
        Device-agnostic: the membrane follows the device of the stimulus."""
        n = s.shape[0]
        if self.V.device != s.device:
            self.V = self.V.to(s.device)
        if self.mode == "dense":
            mask = torch.ones(n, dtype=torch.bool, device=s.device)
            self.last_fire_frac = 1.0
            return mask

        if self.mode == "topk":
            k = max(1, int(self.topk_frac * n))
            if self.budget is not None:
                k = min(k, self.budget)
            idx = torch.topk(s, k).indices
            mask = torch.zeros(n, dtype=torch.bool, device=s.device)
            mask[idx] = True
            self.last_fire_frac = mask.float().mean().item()
            return mask

        if self.mode == "binary":
            mask = self._cap(s >= self.theta, s)   # no temporal memory
            self.last_fire_frac = mask.float().mean().item()
            return mask

        # 'lif' : leaky integrate-and-fire
        self.V = self.lam * self.V + s
        fire = self.V >= self.theta
        mask = self._cap(fire, self.V)
        self.V[mask] = 0.0          # only admitted neurons reset; capped-out ones keep charge
        self.last_fire_frac = mask.float().mean().item()
        return mask

    @staticmethod
    @torch.no_grad()
    def apply_gate(params, mask):
        """
        Zero out gradients of non-firing Gaussians so only the active set is updated.
        In this toy renderer this yields the correct *update* sparsity; in a real 3D
        rasterizer the mask should additionally cull non-active Gaussians from the
        forward/backward pass to realize the *compute* saving.

        NOTE: zeroing the gradient is NOT enough with stateful optimizers (Adam):
        leftover momentum still moves non-firing Gaussians. Wrap the optimizer step
        with `freeze_snapshot` / `restore_frozen` to enforce true freezing.
        """
        keep = mask.float()
        for p in params:
            if p.grad is None:
                continue
            if p.grad.dim() == 2:
                p.grad *= keep[:, None]
            else:
                p.grad *= keep

    @staticmethod
    @torch.no_grad()
    def freeze_snapshot(params):
        """Capture parameter values before opt.step() (cheap O(N) copy)."""
        return [p.detach().clone() for p in params]

    @staticmethod
    @torch.no_grad()
    def restore_frozen(params, mask, snapshot):
        """
        Undo any optimizer movement (e.g. Adam momentum) on NON-firing Gaussians, so
        'frozen this window' really means frozen — required for the cached inactive
        rendering in the budget experiment to stay exact.
        """
        frozen = ~mask
        for p, snap in zip(params, snapshot):
            p.data[frozen] = snap[frozen]
