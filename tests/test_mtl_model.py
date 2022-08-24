"""MTLModel forward shape sanity."""

import torch
import torch.nn as nn

from src.core.mtl_model import MTLModel
from src.core.tasks import Task, make_ce_loss


class _Trunk(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(4, dim)

    def forward(self, x):
        return self.fc(x)


class _Head(nn.Module):
    def __init__(self, dim: int = 8, n: int = 3):
        super().__init__()
        self.fc = nn.Linear(dim, n)

    def forward(self, feat):
        return self.fc(feat)


def _make_model():
    tasks = [
        Task("a", _Head(8, 3), make_ce_loss()),
        Task("b", _Head(8, 5), make_ce_loss()),
    ]
    return MTLModel(_Trunk(8), tasks)


def test_forward_returns_dict_with_task_keys():
    model = _make_model()
    out = model(torch.randn(2, 4))
    assert set(out) == {"a", "b"}


def test_per_task_shapes():
    model = _make_model()
    out = model(torch.randn(2, 4))
    assert out["a"].shape == (2, 3)
    assert out["b"].shape == (2, 5)


def test_task_names_property():
    model = _make_model()
    assert model.task_names == ["a", "b"]


def test_heads_are_module_dict():
    model = _make_model()
    assert isinstance(model.heads, torch.nn.ModuleDict)
    assert "a" in model.heads and "b" in model.heads
