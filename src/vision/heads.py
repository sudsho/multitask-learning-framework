"""Task heads on top of ResNetBackbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPoolClassifierHead(nn.Module):
    """C5 -> avgpool -> linear."""

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.pool(feats).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class FCNDecoderHead(nn.Module):
    """Light FCN-style decoder. C5 -> 1x1 conv -> bilinear upsample to input."""

    def __init__(self, feature_dim: int, num_classes: int, out_size: int = 224):
        super().__init__()
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        self.out_size = out_size

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(feats)
        return F.interpolate(
            logits,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )
