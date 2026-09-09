"""Tests for the LIF gate — the core novelty. Verifies each gating mode and, crucially,
the LEAKY TEMPORAL INTEGRATION property that distinguishes LIF from a memoryless mask."""
from __future__ import annotations
import torch

from evspikegs.lif_gate import LIFGate

N = 32


def test_dense_mask_all_true():
    gate = LIFGate(N, mode="dense")
    mask = gate.step(torch.rand(N))
    assert mask.dtype == torch.bool and mask.all()
    assert gate.last_fire_frac == 1.0


def test_topk_exact_fraction():
    gate = LIFGate(N, mode="topk", topk_frac=0.25)
    s = torch.arange(N, dtype=torch.float32)
    mask = gate.step(s)
    k = int(0.25 * N)
    assert int(mask.sum()) == k
    # the k largest stimuli must be the ones selected
    assert mask[-k:].all() and not mask[:-k].any()


def test_binary_has_no_temporal_memory():
    """A sub-threshold stimulus must NEVER fire a memoryless binary mask, however long."""
    gate = LIFGate(N, mode="binary", theta=0.5)
    weak = torch.full((N,), 0.2)          # persistently below theta
    for _ in range(100):
        mask = gate.step(weak)
        assert not mask.any()


def test_lif_integrates_weak_persistent_stimulus():
    """Weak-but-persistent stimulus accumulates in the membrane and eventually fires —
    the leaky-integration property that motivates the whole mechanism (claim C2)."""
    lam, theta, weak = 0.9, 0.5, 0.2
    # steady-state membrane = weak / (1 - lam) = 2.0 > theta, so it must fire eventually
    gate = LIFGate(N, mode="lif", lam=lam, theta=theta)
    fired_at = None
    for t in range(100):
        mask = gate.step(torch.full((N,), weak))
        if mask.any():
            fired_at = t
            break
    assert fired_at is not None, "LIF never fired on a persistent stimulus"
    assert fired_at > 0, "should take >1 window to integrate a sub-threshold stimulus"


def test_lif_transient_noise_leaks_away():
    """A single noise burst below the steady-state level decays without firing."""
    gate = LIFGate(N, mode="lif", lam=0.5, theta=1.0)
    burst = torch.full((N,), 0.8)          # one-off, below theta
    assert not gate.step(burst).any()
    for _ in range(50):                    # silence afterwards: V must decay, never fire
        assert not gate.step(torch.zeros(N)).any()
    assert gate.V.max() < 1e-3


def test_lif_reset_after_fire():
    gate = LIFGate(N, mode="lif", lam=0.9, theta=0.5)
    mask = gate.step(torch.full((N,), 1.0))    # far above theta -> fire immediately
    assert mask.all()
    assert torch.all(gate.V == 0.0), "fired neurons must reset their membrane"


def test_budget_caps_active_set_and_keeps_charge():
    gate = LIFGate(N, mode="lif", lam=0.9, theta=0.5, budget=5)
    s = torch.linspace(0.6, 1.6, N)            # all above theta
    mask = gate.step(s)
    assert int(mask.sum()) == 5, "hard budget must cap the active set"
    assert mask[-5:].all(), "the highest-membrane neurons are admitted first"
    assert torch.all(gate.V[~mask] > 0), "capped-out neurons keep their charge (backlog)"
    # backlog is served next window even with zero new stimulus
    mask2 = gate.step(torch.zeros(N))
    assert int(mask2.sum()) == 5 and not (mask & mask2).any()


def test_apply_gate_zeroes_non_firing_grads():
    p2 = torch.zeros(N, 2, requires_grad=True); p2.grad = torch.ones(N, 2)
    p1 = torch.zeros(N, requires_grad=True); p1.grad = torch.ones(N)
    mask = torch.zeros(N, dtype=torch.bool); mask[:3] = True
    LIFGate.apply_gate([p2, p1], mask)
    assert p2.grad[:3].abs().sum() > 0 and p2.grad[3:].abs().sum() == 0
    assert p1.grad[:3].abs().sum() > 0 and p1.grad[3:].abs().sum() == 0


def test_freeze_restore_defeats_optimizer_momentum():
    """Adam momentum moves zero-grad params; freeze/restore must undo that drift."""
    p = torch.zeros(N, requires_grad=True)
    opt = torch.optim.Adam([p], lr=0.1)
    mask = torch.zeros(N, dtype=torch.bool); mask[0] = True
    # build momentum on every neuron
    p.grad = torch.ones(N); opt.step()
    p.data.zero_()
    # gated step: only neuron 0 has grad, the rest must stay exactly frozen
    p.grad = torch.zeros(N); p.grad[0] = 1.0
    snap = LIFGate.freeze_snapshot([p])
    opt.step()
    LIFGate.restore_frozen([p], mask, snap)
    assert p.data[1:].abs().max() == 0.0, "frozen neurons moved (momentum leak)"
    assert p.data[0] != 0.0, "the firing neuron must still be updated"


def test_stimulus_from_grads_matches_manual_norm():
    p2 = torch.zeros(N, 3, requires_grad=True); p2.grad = torch.randn(N, 3)
    p1 = torch.zeros(N, requires_grad=True); p1.grad = torch.randn(N)
    s = LIFGate.stimulus_from_grads([p2, p1])
    expected = p2.grad.norm(dim=1) + p1.grad.abs()
    assert torch.allclose(s, expected)
