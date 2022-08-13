"""Vision demo data.

A toy classification + segmentation dataset. We synthesise simple coloured
shapes on a black background so the demo runs without downloading Pascal VOC.
The classification label is the shape class; the segmentation mask is the
per-pixel class.

For the real run, point this at torchvision's VOCSegmentation / VOCDetection
or similar.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


CLS_NAMES = ["background", "circle", "square", "triangle"]
NUM_CLS = 3   # foreground classes only (1..3); 0 is background
NUM_SEG = 4   # incl. background


def _draw_circle(mask: np.ndarray, cx: int, cy: int, r: int, value: int) -> None:
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    sel = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    mask[sel] = value


def _draw_square(mask: np.ndarray, cx: int, cy: int, half: int, value: int) -> None:
    y0, y1 = max(0, cy - half), min(mask.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(mask.shape[1], cx + half)
    mask[y0:y1, x0:x1] = value


def _draw_triangle(mask: np.ndarray, cx: int, cy: int, half: int, value: int) -> None:
    H, W = mask.shape
    for i in range(2 * half):
        width = (i // 2) + 1
        y = cy - half + i
        if 0 <= y < H:
            x0 = max(0, cx - width)
            x1 = min(W, cx + width)
            mask[y, x0:x1] = value


@dataclass
class _Synth:
    image: np.ndarray
    label: int
    seg: np.ndarray


def _make_example(rng: np.random.RandomState, size: int = 64) -> _Synth:
    img = np.zeros((3, size, size), dtype=np.float32)
    seg = np.zeros((size, size), dtype=np.int64)
    shape = rng.randint(1, 4)   # 1..3
    cx, cy = rng.randint(size // 4, 3 * size // 4, size=2)
    r = rng.randint(size // 8, size // 4)

    if shape == 1:
        _draw_circle(seg, cx, cy, r, 1)
    elif shape == 2:
        _draw_square(seg, cx, cy, r, 2)
    else:
        _draw_triangle(seg, cx, cy, r, 3)

    # paint image with a colour per class
    palette = {1: (1.0, 0.2, 0.2), 2: (0.2, 1.0, 0.2), 3: (0.2, 0.2, 1.0)}
    rgb = palette[shape]
    for c in range(3):
        img[c] = (seg > 0).astype(np.float32) * rgb[c]

    return _Synth(img, shape - 1, seg)   # cls in [0..2]


class ToyMTLVisionDataset(Dataset):
    def __init__(self, n: int = 256, image_size: int = 64, seed: int = 0):
        self.image_size = image_size
        rng = np.random.RandomState(seed)
        self.items = [_make_example(rng, image_size) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e = self.items[idx]
        return {
            "image": torch.from_numpy(e.image),
            "classification": torch.tensor(e.label, dtype=torch.long),
            "segmentation": torch.from_numpy(e.seg),
        }


def collate_vision(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    return {
        "inputs": out["image"],
        "targets": {
            "classification": out["classification"],
            "segmentation": out["segmentation"],
        },
    }
