"""Multi-task trainer with optional mlflow logging.

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
        mlflow_logger=None,
    ):
        self.model = model.to(device)
        self.weighter = weighter.to(device)
        self.optimizer = optimizer
        self.device = device
        self.mlflow_logger = mlflow_logger
        self._initial_losses: Optional[Dict[str, float]] = None
        # GradNorm needs an optimizer for the task-weight params alone.
        self._gn_optim: Optional[torch.optim.Optimizer] = None
        if isinstance(weighter, GradNormWeighter):
            self._gn_optim = torch.optim.Adam([weighter.weights], lr=2e-4)

    def _move(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move(v) for k, v in obj.items()}
        return obj

    def _per_task_losses(
        self, batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        preds = self.model(batch["inputs"])
        targets = batch["targets"]
        losses: Dict[str, torch.Tensor] = {}
        for t in self.model.tasks:
            losses[t.name] = t.compute_loss(preds[t.name], targets[t.name])
        return losses

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        batch = self._move(batch)
        self.model.train()
        self.optimizer.zero_grad()

        losses = self._per_task_losses(batch)

        if self._initial_losses is None:
            self._initial_losses = {n: float(v.detach().item())
                                    for n, v in losses.items()}

        total = self.weighter.combine(losses)

        if isinstance(self.weighter, GradNormWeighter):
            # First main backward (retain so we can compute gradnorm grads).
            total.backward(retain_graph=True)
            shared_param = self.model.backbone.last_layer.weight
            self._gn_optim.zero_grad()
            gn_loss = self.weighter.update(losses, self._initial_losses, shared_param)
            gn_loss.backward()
            self._gn_optim.step()
            # Renormalise the task weights so they sum to N (per the paper).
            with torch.no_grad():
                w = self.weighter.weights
                w.data = w.data * (len(w) / w.data.sum().clamp_min(1e-8))
        else:
            total.backward()

        self.optimizer.step()

        out = {
            "total": float(total.detach().item()),
            **{n: float(v.detach().item()) for n, v in losses.items()},
        }
        # Pull current weights from the weighter if it exposes them.
        if hasattr(self.weighter, "get_weights"):
            for k, v in self.weighter.get_weights().items():
                out[f"w/{k}"] = v
        return out

    def fit(
        self,
        dataloader: Iterable,
        epochs: int = 1,
        log_every: int = 50,
        freeze_until_epoch: int = 0,
    ):
        history = []
        step = 0
        for epoch in range(epochs):
            # shared backbone freeze schedule:
            # freeze for the first `freeze_until_epoch` epochs, then unfreeze.
            if hasattr(self.model.backbone, "freeze"):
                if epoch < freeze_until_epoch:
                    self.model.backbone.freeze(True)
                else:
                    self.model.backbone.freeze(False)

            for batch in dataloader:
                metrics = self.train_step(batch)
                if step % log_every == 0:
                    msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                   if isinstance(v, float))
                    print(f"[epoch {epoch} step {step}] {msg}")
                if self.mlflow_logger is not None:
                    self.mlflow_logger.log(metrics, step=step)
                history.append({"step": step, "epoch": epoch, **metrics})
                step += 1
        return history


class MLflowLogger:
    """Thin wrapper so the trainer doesn't import mlflow at module level."""

    def __init__(self, run_name: str, tracking_uri: Optional[str] = None):
        import mlflow

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.run = mlflow.start_run(run_name=run_name)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        self.mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, float)},
            step=step,
        )

    def close(self) -> None:
        self.mlflow.end_run()
