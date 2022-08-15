"""End-to-end NLP MTL demo.

  python examples/run_nlp.py --config configs/nlp.yaml
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

# Allow running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.loss_weighting import (
    GradNormWeighter,
    UncertaintyWeighter,
    UniformWeighter,
)
from src.core.mtl_model import MTLModel
from src.core.tasks import Task, make_ce_loss
from src.core.trainer import MTLTrainer
from src.nlp.bert_backbone import BertBackbone
from src.nlp.data import NUM_NER, PAD_NER_ID, ToyMTLNLPDataset, collate
from src.nlp.heads import SentenceClassifierHead, TokenClassifierHead


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_weighter(strategy: str, task_names, alpha: float = 1.5):
    if strategy == "uniform":
        return UniformWeighter(task_names)
    if strategy == "uncertainty":
        return UncertaintyWeighter(task_names)
    if strategy == "gradnorm":
        return GradNormWeighter(task_names, alpha=alpha)
    raise ValueError(f"unknown strategy {strategy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizerFast.from_pretrained(cfg["backbone"]["pretrained"])
    ds = ToyMTLNLPDataset(tokenizer, max_seq_len=cfg["data"]["max_seq_len"])
    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=cfg["data"]["num_workers"],
    )

    backbone = BertBackbone(cfg["backbone"]["pretrained"])
    H = backbone.hidden_size

    tasks = [
        Task(
            "sentiment",
            SentenceClassifierHead(H, 2),
            make_ce_loss(),
        ),
        Task(
            "topic",
            SentenceClassifierHead(H, 4),
            make_ce_loss(),
        ),
        Task(
            "ner",
            TokenClassifierHead(H, NUM_NER),
            lambda logits, target: torch.nn.functional.cross_entropy(
                logits.view(-1, NUM_NER),
                target.view(-1),
                ignore_index=PAD_NER_ID,
            ),
        ),
    ]
    model = MTLModel(backbone, tasks)
    weighter = build_weighter(
        cfg["loss_weighting"]["strategy"],
        [t.name for t in tasks],
        alpha=cfg["loss_weighting"].get("gradnorm", {}).get("alpha", 1.5),
    )

    optim = torch.optim.AdamW(
        list(model.parameters()) + list(weighter.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    trainer = MTLTrainer(model, weighter, optim, device=device)
    history = trainer.fit(
        loader,
        epochs=cfg["train"]["epochs"],
        log_every=cfg["train"]["log_every"],
    )

    out_dir = cfg["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"history": history}, os.path.join(out_dir, "history.pt"))
    print(f"done. {len(history)} steps. saved to {out_dir}/history.pt")


if __name__ == "__main__":
    main()
