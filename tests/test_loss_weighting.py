"""Tests for the three loss-weighting strategies."""

import torch

from src.core.loss_weighting import (
    GradNormWeighter,
    UncertaintyWeighter,
    UniformWeighter,
)


def _losses(a: float, b: float):
    return {"a": torch.tensor(a, requires_grad=True),
            "b": torch.tensor(b, requires_grad=True)}


def test_uniform_weighter_is_mean():
    w = UniformWeighter(["a", "b"])
    out = w.combine(_losses(2.0, 4.0))
    assert torch.allclose(out, torch.tensor(3.0))


def test_uncertainty_weighter_with_zero_log_sigma_is_sum_plus_zero():
    # exp(-0)=1, log_sigma=0 -> combined = sum of losses
    w = UncertaintyWeighter(["a", "b"], init_log_sigma=0.0)
    out = w.combine(_losses(2.0, 4.0))
    assert torch.allclose(out, torch.tensor(6.0))


def test_uncertainty_weighter_get_weights():
    w = UncertaintyWeighter(["a", "b"], init_log_sigma=0.0)
    weights = w.get_weights()
    assert set(weights) == {"a", "b"}
    for v in weights.values():
        assert abs(v - 1.0) < 1e-6


def test_gradnorm_weighter_combine_initial_unit_weights():
    w = GradNormWeighter(["a", "b"])
    out = w.combine(_losses(2.0, 4.0))
    # weights = 1.0 each, so it's a sum
    assert torch.allclose(out, torch.tensor(6.0))


def test_gradnorm_update_returns_scalar_loss():
    """End-to-end smoke: build a tiny shared param, run an update step."""
    torch.manual_seed(0)
    shared = torch.nn.Parameter(torch.randn(8, requires_grad=True))
    w = GradNormWeighter(["a", "b"], alpha=1.5)

    # Construct task losses that depend on `shared`.
    losses = {
        "a": (shared ** 2).sum() * 1.0,
        "b": ((shared - 0.5) ** 2).sum() * 0.7,
    }
    initial = {"a": float(losses["a"].detach()), "b": float(losses["b"].detach())}

    gn_loss = w.update(losses, initial, shared)
    assert gn_loss.dim() == 0
    assert torch.isfinite(gn_loss)
