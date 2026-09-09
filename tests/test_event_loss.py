"""Tests for the event model + differentiable event photometric loss.
Verifies the hard constraint: supervision lives on LOG-intensity DIFFERENCES only."""
from __future__ import annotations
import torch

from evspikegs.event_data import accumulate_events
from evspikegs.event_loss import event_photometric_loss, anchor_loss


def test_no_change_means_no_events():
    L = torch.randn(100)
    dE = accumulate_events(L, L.clone(), C=0.1)
    assert torch.all(dE == 0.0), "identical log-images must produce zero events"


def test_events_quantized_to_contrast_steps():
    C = 0.1
    L_prev = torch.zeros(5)
    L_curr = torch.tensor([0.0, 0.04, 0.11, -0.26, 0.35])
    dE = accumulate_events(L_prev, L_curr, C)
    # round(dL/C)*C: 0, 0 (below threshold), 0.1, -0.3, 0.4 (allow fp wiggle)
    assert torch.allclose(dE, torch.tensor([0.0, 0.0, 0.1, -0.3, 0.4]), atol=1e-6)
    steps = dE / C
    assert torch.allclose(steps, steps.round(), atol=1e-5), "dE must be integer multiples of C"


def test_polarity_sign():
    C = 0.2
    brighter = accumulate_events(torch.zeros(1), torch.full((1,), 0.5), C)
    darker = accumulate_events(torch.zeros(1), torch.full((1,), -0.5), C)
    assert brighter.item() > 0 and darker.item() < 0


def test_loss_zero_at_perfect_prediction_and_decreases_towards_it():
    dE = torch.tensor([0.1, -0.2, 0.0, 0.3])
    assert event_photometric_loss(dE.clone(), dE).item() < 1e-12
    far = event_photometric_loss(dE + 1.0, dE).item()
    near = event_photometric_loss(dE + 0.1, dE).item()
    assert far > near > 0.0, "loss must shrink as dLhat approaches dE"


def test_loss_gradient_descends_to_target():
    """A few gradient steps on the loss alone must drive dLhat toward dE."""
    dE = torch.tensor([0.2, -0.1, 0.4])
    dLhat = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([dLhat], lr=0.05)
    first = event_photometric_loss(dLhat, dE).item()
    for _ in range(200):
        opt.zero_grad()
        loss = event_photometric_loss(dLhat, dE)
        loss.backward()
        opt.step()
    assert event_photometric_loss(dLhat, dE).item() < 0.1 * first


def test_anchor_loss_is_l1():
    a, b = torch.tensor([0.0, 1.0]), torch.tensor([0.5, 0.5])
    assert abs(anchor_loss(a, b).item() - 0.5) < 1e-6
