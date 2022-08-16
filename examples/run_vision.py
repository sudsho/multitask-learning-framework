"""End-to-end vision MTL demo (classification + segmentation).

  python examples/run_vision.py --config configs/vision.yaml
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.loss_weighting import (
    GradNormWeighter,
    UncertaintyWeighter,
    UniformWeighter,
)
from src.core.mtl_model import MTLModel
from src.core.tasks import Task, make_ce_loss
from src.core.trainer import MTLTrainer
from src.vision.data import NUM_CLS, NUM_SEG, ToyMTLVisionDataset, collate_vision
from src.vision.heads import AvgPoolClassifierHead, FCNDecoderHead
from src.vision.resnet_backbone import ResNetBackbone


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_weighter(strategy, names, alpha: float = 1.5):
    if strategy == "uniform":
        return UniformWeighter(names)
    if strategy == "uncertainty":
        return UncertaintyWeighter(names)
    if strategy == "gradnorm":
        return GradNormWeighter(names, alpha=alpha)
    raise ValueError(f"unknown strategy {strategy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = cfg["data"]["image_size"]
    ds = ToyMTLVisionDataset(n=128, image_size=img_size)
    loader = DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_vision,
        num_workers=cfg["data"]["num_workers"],
    )

    backbone = ResNetBackbone(
        variant=cfg["backbone"]["type"],
        pretrained=cfg["backbone"]["pretrained"],
    )
    F = backbone.feature_dim

    tasks = [
        Task(
            "classification",
            AvgPoolClassifierHead(F, NUM_CLS),
            make_ce_loss(),
        ),
        Task(
            "segmentation",
            FCNDecoderHead(F, NUM_SEG, out_size=img_size),
            make_ce_loss(ignore_index=255),
        ),
    ]
    model = MTLModel(backbone, tasks)
    weighter = build_weighter(
        cfg["loss_weighting"]["strategy"],
        [t.name for t in tasks],
        alpha=cfg["loss_weighting"].get("gradnorm", {}).get("alpha", 1.5),
    )

    optim = torch.optim.SGD(
        list(model.parameters()) + list(weighter.parameters()),
        lr=float(cfg["train"]["lr"]),
        momentum=cfg["train"]["momentum"],
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
