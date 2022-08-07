"""ResNet shared backbone.

Returns the C5 feature map (before global average pool). The classifier head
adds avgpool + fc; the segmentation head upsamples C5 to image resolution.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetBackbone(nn.Module):
    def __init__(self, variant: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if variant == "resnet18":
            net = tvm.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif variant == "resnet50":
            net = tvm.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"unknown resnet variant: {variant}")

        # Strip avgpool + fc; we keep the conv trunk only.
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    @property
    def last_layer(self) -> nn.Module:
        # Used by GradNorm. Picks the last conv before classification.
        return self.layer4[-1].conv3 if hasattr(self.layer4[-1], "conv3") else self.layer4[-1].conv2

    def freeze(self, freeze: bool = True) -> None:
        for p in self.parameters():
            p.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B, C, H/32, W/32)
        return x
