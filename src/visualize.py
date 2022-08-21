"""Inspection / visualisation tools for MTL training.

Inputs are the `history` lists returned by `MTLTrainer.fit`. Each entry is
a dict like:
    {"step": int, "epoch": int, "total": float,
     "<task>": float, "w/<task>": float, ...}
"""

from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt


def _series(history: Sequence[Dict], key: str) -> List[float]:
    return [h[key] for h in history if key in h]


def plot_per_task_losses(history, task_names: Iterable[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in task_names:
        ax.plot(_series(history, n), label=n)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("per-task loss curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_task_weights(history, task_names: Iterable[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for n in task_names:
        ys = _series(history, f"w/{n}")
        if ys:
            ax.plot(ys, label=n)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no per-task weights logged", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("step")
    ax.set_ylabel("weight")
    ax.set_title("task weights over training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_total_loss(history, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(_series(history, "total"), color="black")
    ax.set_xlabel("step")
    ax.set_ylabel("combined loss")
    ax.set_title("combined MTL loss")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gradnorm_panel(history, task_names, path: str) -> None:
    """Three-panel summary: loss, weights, total. Useful when running gradnorm."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for n in task_names:
        axes[0].plot(_series(history, n), label=n)
        ys = _series(history, f"w/{n}")
        if ys:
            axes[1].plot(ys, label=n)
    axes[0].set_title("per-task loss")
    axes[0].legend()
    axes[1].set_title("task weights")
    axes[1].legend()
    axes[2].plot(_series(history, "total"), color="black")
    axes[2].set_title("combined")
    for ax in axes:
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
