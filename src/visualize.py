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
