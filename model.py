"""Model definitions for distributed training.

Contains a lightweight ResNet-variant (``MiniResNet``) suitable for
image classification demos and a factory that wraps any model with
``DistributedDataParallel``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainingConfig


# ── Building blocks ─────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Two-conv residual block with optional down-sampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# ── Main model ──────────────────────────────────────────────────────

class MiniResNet(nn.Module):
    """Compact ResNet (3 stages, 2 blocks each) for image classification.

    Architecture
    ------------
    stem  → [64] stage-1 → [128] stage-2 → [256] stage-3 → GAP → FC
    """

    _STAGE_CHANNELS = (64, 128, 256)
    _BLOCKS_PER_STAGE = 2

    def __init__(self, in_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        layers: list[nn.Module] = []
        prev_ch = 64
        for ch in self._STAGE_CHANNELS:
            stride = 1 if ch == self._STAGE_CHANNELS[0] else 2
            layers.append(ResidualBlock(prev_ch, ch, stride=stride))
            for _ in range(1, self._BLOCKS_PER_STAGE):
                layers.append(ResidualBlock(ch, ch))
            prev_ch = ch
        self.stages = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._STAGE_CHANNELS[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ── Factories ───────────────────────────────────────────────────────

def create_model(config: TrainingConfig, device: torch.device) -> nn.Module:
    """Instantiate a ``MiniResNet`` and move it to *device*."""
    model = MiniResNet(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
    )
    return model.to(device)


def wrap_ddp(model: nn.Module, local_rank: int) -> DDP:
    """Wrap *model* with ``DistributedDataParallel``."""
    return DDP(model, device_ids=[local_rank])
