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


class UncertaintyWeighter(BaseWeighter):
    """Learnable homoscedastic uncertainty weighting.

    For each task we keep a learnable `log_sigma`. The combined loss is
        sum_i ( exp(-log_sigma_i) * loss_i + log_sigma_i )

    which is the negative log likelihood of independent Gaussians (regression)
    or the relaxed multi-class form for classification (Kendall et al 2018).
    Adding `log_sigma_i` keeps the optimizer from collapsing weights to zero.
    """

    def __init__(self, task_names: List[str], init_log_sigma: float = 0.0):
        super().__init__(task_names)
        self.log_sigma = nn.Parameter(
            torch.full((len(task_names),), float(init_log_sigma))
        )

    def combine(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, name in enumerate(self.task_names):
            ls = self.log_sigma[i]
            total = total + torch.exp(-ls) * losses[name] + ls
        return total

    def get_weights(self) -> Dict[str, float]:
        with torch.no_grad():
            return {
                n: float(torch.exp(-self.log_sigma[i]).item())
                for i, n in enumerate(self.task_names)
            }


class GradNormWeighter(BaseWeighter):
    """GradNorm (Chen et al 2018).

    Keeps a learnable weight w_i per task and adjusts them so the L2 grad norms
    of `w_i * L_i` against a shared parameter tensor stay close to a target
    that depends on each task's relative training rate.

    Note: this implementation expects the trainer to call
        `update(losses, initial_losses, shared_param)`
    AFTER the main weighted loss has done its backward, but BEFORE the
    optimizer step.
    """

    def __init__(self, task_names: List[str], alpha: float = 1.5):
        super().__init__(task_names)
        # Initialise all task weights to 1.
        self.weights = nn.Parameter(torch.ones(len(task_names)))
        self.alpha = alpha

    def combine(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, n in enumerate(self.task_names):
            total = total + self.weights[i] * losses[n]
        return total

    def get_weights(self) -> Dict[str, float]:
        with torch.no_grad():
            return {n: float(self.weights[i].item())
                    for i, n in enumerate(self.task_names)}
