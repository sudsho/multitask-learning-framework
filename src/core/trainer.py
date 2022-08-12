"""Multi-task trainer.

Loop:
  for batch in dataloader:
      preds = model(batch.inputs)                    # dict[task_name -> tensor]
      losses = {t.name: t.compute_loss(preds[t.name], batch.targets[t.name])
                for t in model.tasks}                 # dict[task_name -> scalar]
      total = weighter.combine(losses)               # scalar
      total.backward()
      if isinstance(weighter, GradNormWeighter):
          # extra step: update task weights
          ...
      optimizer.step()
"""

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

from .loss_weighting import BaseWeighter, GradNormWeighter
from .mtl_model import MTLModel


class MTLTrainer:
    def __init__(
        self,
        model: MTLModel,
        weighter: BaseWeighter,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.weighter = weighter.to(device)
        self.optimizer = optimizer
        self.device = device
        self._initial_losses: Optional[Dict[str, float]] = None
