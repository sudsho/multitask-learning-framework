"""Base abstractions for a task in the MTL setting.

A Task bundles three things:
  - a name (used as key in batches and head dict)
  - a head module (how the shared features get turned into task outputs)
  - a loss function

The trainer is task-agnostic; it iterates over registered tasks and treats them
uniformly. Adding a new task means subclassing `Task` (or just instantiating it
with the right args) and registering it in the model.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class Task:
    name: str
    head: nn.Module
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    # Optional metadata: number of classes, head type, anything else.
    meta: Dict[str, Any] = field(default_factory=dict)
    # Initial loss weight. Used by uniform weighter and as init for others.
    weight_init: float = 1.0

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, target)


def make_ce_loss(ignore_index: Optional[int] = None) -> Callable:
    if ignore_index is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
