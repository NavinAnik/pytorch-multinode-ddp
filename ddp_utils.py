"""Distributed-training lifecycle helpers.

Provides setup / teardown of the NCCL process group, a lightweight
``DistributedContext`` carrier, and checkpoint save / load utilities
that are rank-aware.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainingConfig

logger = logging.getLogger(__name__)


# ── Distributed context ─────────────────────────────────────────────

@dataclass(frozen=True)
class DistributedContext:
    """Immutable snapshot of the current process's distributed identity."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def setup_distributed(config: TrainingConfig) -> DistributedContext:
    """Initialize the NCCL process group and pin the correct GPU.

    Reads ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` from the
    environment (set automatically by ``torchrun``).

    Returns
    -------
    DistributedContext
        Snapshot of rank, local_rank, world_size, and assigned device.

    Raises
    ------
    RuntimeError
        If required environment variables are missing or CUDA is
        unavailable.
    """
    required_vars = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [v for v in required_vars if v not in os.environ]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {missing}. "
            "Launch with `torchrun` or set them manually."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for NCCL-based distributed training."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend=config.backend)

    logger.info(
        "Process group initialized  |  rank=%d  local_rank=%d  "
        "world_size=%d  device=%s",
        rank, local_rank, world_size, device,
    )

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed() -> None:
    """Destroy the process group (safe to call even if not initialized)."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Return *True* on the global rank-0 process."""
    return rank == 0


# ── Checkpointing ───────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    epoch: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
    rank: int = 0,
) -> None:
    """Persist training state to disk (rank-0 only).

    Parameters
    ----------
    path : str
        Destination file path.
    epoch : int
        Current epoch number (1-indexed).
    model : DDP
        The wrapped DDP model; ``.module`` is accessed for the raw
        state dict.
    optimizer : Optimizer
        Optimizer whose state is saved.
    scaler : GradScaler, optional
        AMP grad scaler (omitted from checkpoint when *None*).
    extra : dict, optional
        Arbitrary extra entries to store.
    rank : int
        Only rank 0 actually writes.
    """
    if not is_main_process(rank):
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if extra:
        state.update(extra)

    torch.save(state, path)
    logger.info("Checkpoint saved → %s  (epoch %d)", path, epoch)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Restore training state from a checkpoint file.

    All ranks load the checkpoint so that model parameters and
    optimizer buffers are consistent across the group.

    Parameters
    ----------
    path : str
        Path to a ``torch.save``-d checkpoint.
    model : nn.Module
        The **unwrapped** model (before DDP wrapping).
    optimizer : Optimizer, optional
        If provided, its state is restored.
    scaler : GradScaler, optional
        If provided and the checkpoint contains scaler state, restores
        it.
    device : torch.device
        Map location for ``torch.load``.

    Returns
    -------
    int
        The epoch stored in the checkpoint (training resumes from
        epoch + 1).
    """
    dist.barrier()

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    epoch: int = ckpt.get("epoch", 0)
    logger.info("Checkpoint loaded ← %s  (epoch %d)", path, epoch)
    return epoch
