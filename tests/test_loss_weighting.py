"""Tests for the loss-weighting strategies."""

import torch

from src.core.loss_weighting import (
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
