"""Shared-backbone multi-task model.

Sketch:
  features = backbone(inputs)
  outputs  = {task_name: head(features) for each task}
"""

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from .tasks import Task


class MTLModel(nn.Module):
    """Wraps a shared backbone and a `nn.ModuleDict` of per-task heads."""

    def __init__(self, backbone: nn.Module, tasks: Iterable[Task]):
        super().__init__()
        self.backbone = backbone
        self.tasks: List[Task] = list(tasks)
        self.heads = nn.ModuleDict({t.name: t.head for t in self.tasks})

    @property
    def task_names(self) -> List[str]:
        return [t.name for t in self.tasks]

    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        feats = self.backbone(inputs)
        return {name: head(feats) for name, head in self.heads.items()}
