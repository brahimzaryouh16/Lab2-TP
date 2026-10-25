"""Model zoo for Lab 2: scratch CNN, scratch ViT, Faster R-CNN, VGG16/AlexNet."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (AlexNet_Weights, VGG16_Weights,
                                alexnet, vgg16)
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# --------------------------------------------------------------------------
# Part 1.1 — CNN from scratch
# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """VGG-style CNN: 4 conv layers, batch norm, 2 pools, 2 FC layers."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------
# Part 2 — Vision Transformer from scratch
# --------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Block(nn.Module):
    """Pre-norm transformer block (LayerNorm -> attention -> residual)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float,
                 dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, return_attn: bool = False):
        h = self.norm1(x)
        attn_out, weights = self.attn(h, h, h, need_weights=return_attn,
                                      average_attn_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_attn:
            return x, weights
        return x


class ViT(nn.Module):
    """Minimal ViT: patch embedding + CLS token + learned positions + blocks.

    Defaults (patch 4, dim 128, depth 6, 8 heads) give ~1.3M parameters,
    comparable to the scratch CNN, so the comparison is about inductive bias
    rather than capacity.
    """

    def __init__(self, img_size: int = 28, patch_size: int = 4,
                 in_chans: int = 1, num_classes: int = 10, dim: int = 128,
                 depth: int = 6, heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid ** 2
        self.patch_embed = nn.Conv2d(in_chans, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attn: bool = False):
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, N, D
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        attns = []
        for block in self.blocks:
            if return_attn:
                x, w = block(x, return_attn=True)
                attns.append(w)
            else:
                x = block(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        if return_attn:
            return logits, attns
        return logits


# --------------------------------------------------------------------------
# Part 1.4 — ImageNet-pretrained backbones, fine-tuned on MNIST
# --------------------------------------------------------------------------
def build_vgg16(num_classes: int = 10, pretrained: bool = True):
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,
                                    num_classes)
    return model


def build_alexnet(num_classes: int = 10, pretrained: bool = True):
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,
                                    num_classes)
    return model


# --------------------------------------------------------------------------
# Part 1.2 — Faster R-CNN (MobileNetV3 backbone, COCO-pretrained)
# --------------------------------------------------------------------------
def build_frcnn(num_classes: int = 11, pretrained: bool = True,
                min_size: int = 128, max_size: int = 128):
    """Faster R-CNN with a MobileNetV3-Large FPN backbone.

    num_classes includes background, so 11 for digits 1..10.
    """
    weights = (FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
               if pretrained else None)
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, min_size=min_size, max_size=max_size,
        box_score_thresh=0.05, box_nms_thresh=0.5,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)
    return model


CLASSIFIERS = {
    "cnn": lambda **kw: SmallCNN(**kw),
    "vit": lambda **kw: ViT(**kw),
    "vgg16": lambda **kw: build_vgg16(**kw),
    "alexnet": lambda **kw: build_alexnet(**kw),
}

PRETRAINED = {"vgg16", "alexnet", "frcnn"}
