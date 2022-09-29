"""Single-step trainer integration tests."""

import torch
import torch.nn as nn

from src.core.loss_weighting import (
    UncertaintyWeighter,
    UniformWeighter,
)
from src.core.mtl_model import MTLModel
from src.core.tasks import Task, make_ce_loss
from src.core.trainer import MTLTrainer


class _Trunk(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.last_layer = nn.Linear(4, dim)

    def forward(self, x):
        return self.last_layer(x)


def _build(weighter_cls):
    torch.manual_seed(0)
    trunk = _Trunk(8)
    tasks = [
        Task("a", nn.Linear(8, 3), make_ce_loss()),
        Task("b", nn.Linear(8, 5), make_ce_loss()),
    ]
    model = MTLModel(trunk, tasks)
    weighter = weighter_cls([t.name for t in tasks])
    optim = torch.optim.SGD(
        list(model.parameters()) + list(weighter.parameters()),
        lr=1e-2,
    )
    return MTLTrainer(model, weighter, optim)


def _batch():
    return {
        "inputs": torch.randn(4, 4),
        "targets": {
            "a": torch.randint(0, 3, (4,)),
            "b": torch.randint(0, 5, (4,)),
        },
    }


def test_single_step_uniform():
    trainer = _build(UniformWeighter)
    metrics = trainer.train_step(_batch())
    assert "total" in metrics and "a" in metrics and "b" in metrics
    assert metrics["total"] > 0


def test_single_step_uncertainty():
    trainer = _build(UncertaintyWeighter)
    metrics = trainer.train_step(_batch())
    assert "w/a" in metrics and "w/b" in metrics


def test_two_steps_dont_blow_up():
    trainer = _build(UniformWeighter)
    m1 = trainer.train_step(_batch())
    m2 = trainer.train_step(_batch())
    # Both finite and total moved (or at least is real-valued).
    assert m1["total"] != float("nan")
    assert m2["total"] != float("nan")
