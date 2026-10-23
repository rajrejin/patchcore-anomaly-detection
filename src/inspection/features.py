"""Pretrained CNN feature extraction: frozen backbone, layer2 + layer3 patch descriptors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

_WEIGHTS = {
    "resnet18": "ResNet18_Weights",
    "resnet50": "ResNet50_Weights",
    "wide_resnet50_2": "Wide_ResNet50_2_Weights",
}


class RandomWeightsError(RuntimeError):
    """Raised when a train or evaluate run is attempted with random weights."""


def build_backbone(name: str, weights: str) -> tuple[Any, dict[str, str]]:
    """Return a frozen eval-mode torchvision backbone and a weights identifier record."""
    if name not in _WEIGHTS:
        raise ValueError(f"unknown backbone {name}")
    ctor = getattr(torchvision.models, name)
    if weights == "random":
        model = ctor(weights=None)
        info = {"id": "random", "sha256": "none"}
    elif weights == "imagenet":
        enum = getattr(torchvision.models, _WEIGHTS[name]).IMAGENET1K_V1
        model = ctor(weights=enum)
        fname = Path(enum.url).name
        cached = Path(torch.hub.get_dir()) / "checkpoints" / fname
        sha = hashlib.sha256(cached.read_bytes()).hexdigest() if cached.exists() else "unknown"
        info = {"id": f"{name}:IMAGENET1K_V1", "sha256": sha}
    else:
        raise ValueError(f"weights must be 'imagenet' or 'random', got {weights}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, info


def require_real_weights(weights: str) -> None:
    if weights != "imagenet":
        raise RandomWeightsError("train and evaluate refuse random weights (backbone.weights)")


class PatchFeatureExtractor(nn.Module):
    """ResNet up to layer3. Outputs raw layer maps or fused patch descriptors."""

    def __init__(
        self,
        name: str = "resnet50",
        weights: str = "imagenet",
        pool_kernel: int = 3,
        reduce_dim: int | None = None,
    ) -> None:
        super().__init__()
        net, self.weights_info = build_backbone(name, weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.pool_kernel, self.reduce_dim = pool_kernel, reduce_dim
        self.eval()

    def train(self, mode: bool = True) -> PatchFeatureExtractor:  # always eval (frozen BN)
        return super().train(False)

    @torch.no_grad()
    def layer_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(next(self.parameters()).device)  # inputs follow the module's device
        x = self.layer1(self.stem(x))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3

    def fuse(self, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        """Local aggregation, upsample layer3, concatenate, optional channel reduction.
        Returns (B, D, H, W)."""
        k = self.pool_kernel
        if k > 1:
            f2 = F.avg_pool2d(f2, k, stride=1, padding=k // 2)
            f3 = F.avg_pool2d(f3, k, stride=1, padding=k // 2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([f2, f3], dim=1)
        if self.reduce_dim is not None and self.reduce_dim != fused.shape[1]:
            b, c, h, w = fused.shape
            flat = fused.permute(0, 2, 3, 1).reshape(-1, 1, c)
            flat = F.adaptive_avg_pool1d(flat, self.reduce_dim).squeeze(1)
            fused = flat.reshape(b, h, w, self.reduce_dim).permute(0, 3, 1, 2)
        return fused

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2, f3 = self.layer_maps(x)
        return self.fuse(f2, f3)

    @torch.no_grad()
    def descriptors(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H*W, D) patch descriptors in row-major spatial order."""
        fused = self.forward(x)
        return fused.flatten(2).transpose(1, 2).contiguous()


def global_embedding_extractor(name: str, weights: str) -> nn.Module:
    """Stage-1 cascade backbone: global-average-pooled last conv block embedding."""
    net, _ = build_backbone(name, weights)
    net.fc = nn.Identity()
    return net
