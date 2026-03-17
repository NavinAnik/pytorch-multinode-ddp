"""Rank-aware structured logging.

Only rank 0 emits INFO-level messages; all other ranks are silenced to
WARNING so that multi-process output stays clean and readable.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

import torch


def setup_logger(
    rank: int,
    name: str = "ddp",
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a rank-aware logger.

    Parameters
    ----------
    rank : int
        Global rank of the calling process.
    name : str
        Logger name.
    log_level : int
        Effective level for rank-0; other ranks get ``WARNING``.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    effective_level = log_level if rank == 0 else logging.WARNING

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(effective_level)

    fmt = logging.Formatter(
        f"[Rank {rank}] %(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)

    logger.setLevel(effective_level)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_gpu_info(logger: logging.Logger, device: torch.device) -> None:
    """Log GPU name and current memory usage."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — skipping GPU info.")
        return

    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    alloc = torch.cuda.memory_allocated(idx) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)

    logger.info(
        "GPU %d: %s  |  allocated=%.1f MiB  reserved=%.1f MiB",
        idx, name, alloc, reserved,
    )


def log_training_step(
    logger: logging.Logger,
    epoch: int,
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    throughput: Optional[float] = None,
) -> None:
    """Emit a structured per-step log line."""
    parts = [
        f"Epoch [{epoch}]",
        f"Step [{step}/{total_steps}]",
        f"Loss={loss:.4f}",
        f"LR={lr:.6f}",
    ]
    if throughput is not None:
        parts.append(f"Throughput={throughput:.1f} samples/s")
    logger.info("  ".join(parts))


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    metrics: Dict[str, Any],
) -> None:
    """Emit an end-of-epoch summary."""
    items = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())
    logger.info("─── Epoch %d Summary ───  %s", epoch, items)
