"""Loss-weighting strategies for multi-task learning.

Three options:
  1. UniformWeighter:    sum_i (1/N) * loss_i
  2. UncertaintyWeighter: learnable log-sigma per task (Kendall et al 2018)
  3. GradNormWeighter:   adapts weights so grad norms across tasks stay balanced
                          (Chen et al 2018)

All weighters expose `combine(losses: Dict[str, Tensor]) -> Tensor` returning
a single scalar to backward through. GradNorm has an additional `update` hook
that the trainer calls after the main backward pass.
"""

from typing import Dict, List

import torch
import torch.nn as nn


class BaseWeighter(nn.Module):
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = list(task_names)

    def combine(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class UniformWeighter(BaseWeighter):
    """Plain mean of task losses."""

    def combine(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        vals = [losses[n] for n in self.task_names]
        return torch.stack(vals).mean()
