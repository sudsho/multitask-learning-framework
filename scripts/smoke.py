"""Tiny-CPU offline smoke for the multi-task learning framework.

Proves the core MTL machinery end to end without a GPU, without downloading any
pretrained weights (no BERT, no ResNet), and without touching the network:

  * a tiny from-scratch shared encoder (a small MLP trunk),
  * three per-task heads hung off that shared trunk via the same MTLModel /
    nn.ModuleDict used by the real demos: a multi-class classifier, a scalar
    regressor, and a binary classifier,
  * tiny synthetic multi-task data where every target is a learnable function of
    one shared latent, so the shared trunk actually has something to learn,
  * a few real training steps through the real MTLTrainer with the real
    UncertaintyWeighter (Kendall et al 2018) combining a cross-entropy loss and
    an MSE loss into one weighted scalar,
  * inference producing per-task outputs of the right shape.

The full NLP / CV demos (examples/run_nlp.py, examples/run_vision.py) need a GPU,
pretrained BERT / ResNet weights, and real datasets. This smoke deliberately uses
none of that. Run it with:

    python scripts/smoke.py
    make smoke
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Force CPU and single-threaded determinism; this smoke never uses a GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_num_threads(1)

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.loss_weighting import UncertaintyWeighter  # noqa: E402
from src.core.mtl_model import MTLModel  # noqa: E402
from src.core.tasks import Task, make_ce_loss  # noqa: E402
from src.core.trainer import MTLTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# tiny from-scratch shared encoder (no torchvision, no transformers, no download)
# ---------------------------------------------------------------------------
class TinyEncoder(nn.Module):
    """A small MLP trunk standing in for the BERT / ResNet shared backbone."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """Shared features -> single scalar."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc = nn.Linear(hidden, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats).squeeze(-1)


def make_mse_loss():
    mse = nn.MSELoss()
    return lambda pred, target: mse(pred, target)


# ---------------------------------------------------------------------------
# tiny synthetic multi-task data with genuinely learnable shared signal
# ---------------------------------------------------------------------------
def make_synthetic(n, in_dim, n_classes, seed=0):
    """One shared input feeds three tasks whose targets are functions of it.

    A latent input X drives:
      * category:  argmax of a fixed linear projection  (multi-class, CE)
      * value:     a fixed linear function + small noise (regression, MSE)
      * positive:  sign of another fixed projection      (binary, CE)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, in_dim).astype(np.float32)

    w_cls = rng.randn(in_dim, n_classes).astype(np.float32)
    category = (X @ w_cls).argmax(axis=1).astype(np.int64)

    w_reg = rng.randn(in_dim).astype(np.float32)
    value = (X @ w_reg + 0.05 * rng.randn(n)).astype(np.float32)

    w_bin = rng.randn(in_dim).astype(np.float32)
    positive = ((X @ w_bin) > 0).astype(np.int64)

    return {
        "inputs": torch.from_numpy(X),
        "targets": {
            "category": torch.from_numpy(category),
            "value": torch.from_numpy(value),
            "positive": torch.from_numpy(positive),
        },
    }


def batches(data, batch_size):
    """Yield mini-batches in the {inputs, targets} shape the trainer expects."""
    n = data["inputs"].shape[0]
    for s in range(0, n, batch_size):
        e = s + batch_size
        yield {
            "inputs": data["inputs"][s:e],
            "targets": {k: v[s:e] for k, v in data["targets"].items()},
        }


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cpu"
    in_dim, hidden, n_classes = 16, 32, 4
    n_train, batch_size, epochs = 256, 32, 12

    print("Multi-task learning framework: tiny-CPU offline smoke")
    print(f"device={device} (pinned) torch={torch.__version__} "
          f"cuda_available={torch.cuda.is_available()} (ignored)")
    print("shared TinyEncoder trunk + 3 heads "
          "(category=classification, value=regression, positive=binary)\n")

    data = make_synthetic(n_train, in_dim, n_classes, seed=0)

    backbone = TinyEncoder(in_dim, hidden)
    tasks = [
        Task("category", nn.Linear(hidden, n_classes), make_ce_loss()),
        Task("value", RegressionHead(hidden), make_mse_loss()),
        Task("positive", nn.Linear(hidden, 2), make_ce_loss()),
    ]
    model = MTLModel(backbone, tasks)

    # Learnable homoscedastic-uncertainty task weighting (Kendall et al 2018).
    weighter = UncertaintyWeighter([t.name for t in tasks])

    optim = torch.optim.Adam(
        list(model.parameters()) + list(weighter.parameters()), lr=1e-2
    )
    trainer = MTLTrainer(model, weighter, optim, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params} (tiny; runs in well under a second on CPU)\n")

    first, last = None, None
    for epoch in range(epochs):
        epoch_metrics = [trainer.train_step(b) for b in batches(data, batch_size)]
        # average the per-task losses across the epoch's steps
        agg = {k: float(np.mean([m[k] for m in epoch_metrics]))
               for k in ("total", "category", "value", "positive")}
        w = {k: float(np.mean([m[f"w/{k}"] for m in epoch_metrics]))
             for k in ("category", "value", "positive")}
        if epoch == 0:
            first = agg
        last = agg
        print(f"epoch {epoch:2d} | weighted_total={agg['total']:.4f} "
              f"| category(CE)={agg['category']:.4f} "
              f"value(MSE)={agg['value']:.4f} "
              f"positive(CE)={agg['positive']:.4f} "
              f"| weights c={w['category']:.2f} v={w['value']:.2f} p={w['positive']:.2f}")

    print("\nper-task loss change (first epoch -> last epoch):")
    for k in ("category", "value", "positive"):
        print(f"  {k:9s}: {first[k]:.4f} -> {last[k]:.4f}")
    print(f"  weighted total: {first['total']:.4f} -> {last['total']:.4f}")

    # every task loss should have gone down
    improved = all(last[k] < first[k] for k in ("category", "value", "positive"))

    # ---- inference: per-task outputs of the right shape ----
    model.eval()
    with torch.no_grad():
        infer = make_synthetic(5, in_dim, n_classes, seed=1)
        out = model(infer["inputs"])
    print("\ninference on 5 fresh examples -> per-task output shapes:")
    print(f"  category  (logits) : {tuple(out['category'].shape)}  expected (5, {n_classes})")
    print(f"  value     (scalar) : {tuple(out['value'].shape)}  expected (5,)")
    print(f"  positive  (logits) : {tuple(out['positive'].shape)}  expected (5, 2)")

    shapes_ok = (
        tuple(out["category"].shape) == (5, n_classes)
        and tuple(out["value"].shape) == (5,)
        and tuple(out["positive"].shape) == (5, 2)
    )

    ok = improved and shapes_ok
    print("\nSMOKE PASS" if ok else "\nSMOKE FAIL")
    if not improved:
        print("  (a task loss did not decrease)")
    if not shapes_ok:
        print("  (an inference output had the wrong shape)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
