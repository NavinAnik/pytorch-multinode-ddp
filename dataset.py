"""Dataset and distributed data-loading utilities.

Provides a synthetic image-classification dataset (no downloads
required) and a factory that builds a ``DataLoader`` with a
``DistributedSampler``.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config import TrainingConfig
from ddp_utils import DistributedContext


class SyntheticImageDataset(Dataset):
    """Random image/label pairs for pipeline testing.

    Generates all tensors lazily per ``__getitem__`` so memory stays
    constant regardless of *size*.

    Parameters
    ----------
    size : int
        Number of samples in the dataset.
    image_size : int
        Height and width of each square image.
    in_channels : int
        Number of colour channels.
    num_classes : int
        Label range ``[0, num_classes)``.
    """

    def __init__(
        self,
        size: int = 50_000,
        image_size: int = 32,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        self.size = size
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        generator = torch.Generator().manual_seed(index)
        image = torch.randn(
            self.in_channels, self.image_size, self.image_size,
            generator=generator,
        )
        label = torch.randint(
            0, self.num_classes, (1,), generator=generator,
        ).item()
        return image, int(label)


def create_distributed_dataloader(
    dataset: Dataset,
    config: TrainingConfig,
    ctx: DistributedContext,
) -> Tuple[DataLoader, DistributedSampler]:
    """Build a ``DataLoader`` backed by a ``DistributedSampler``.

    Returns the loader **and** the sampler so callers can call
    ``sampler.set_epoch(epoch)`` at the start of each epoch.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    return loader, sampler
