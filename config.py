"""Centralized training configuration.

All tunable hyperparameters and runtime settings live here so that no
module in the project contains hard-coded magic numbers.  The config is
built from CLI arguments merged on top of dataclass defaults.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Immutable bag of every parameter the training pipeline needs."""

    # ── Model ────────────────────────────────────────────────────────
    num_classes: int = 10
    in_channels: int = 3
    image_size: int = 32

    # ── Data ─────────────────────────────────────────────────────────
    data_dir: str = "./data"
    dataset_size: int = 50_000
    batch_size: int = 64  # per-GPU
    num_workers: int = 4

    # ── Optimizer / Scheduler ────────────────────────────────────────
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # ── Distributed ──────────────────────────────────────────────────
    backend: str = "nccl"

    # ── Mixed Precision ──────────────────────────────────────────────
    use_amp: bool = True

    # ── Gradient Accumulation ────────────────────────────────────────
    grad_accum_steps: int = 1

    # ── Checkpointing ────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1  # save every N epochs
    resume_from: Optional[str] = None

    # ── Logging ──────────────────────────────────────────────────────
    log_interval: int = 10  # log every N steps

    # ── Profiling ────────────────────────────────────────────────────
    enable_profiling: bool = False

    # ── Seed ─────────────────────────────────────────────────────────
    seed: int = 42

    @classmethod
    def from_args(cls) -> "TrainingConfig":
        """Parse CLI flags and return a populated config."""
        parser = argparse.ArgumentParser(
            description="Multi-Node DDP Training Pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        defaults = cls()
        for name, val in vars(defaults).items():
            arg_type = type(val) if val is not None else str
            if isinstance(val, bool):
                parser.add_argument(
                    f"--{name}",
                    default=val,
                    action=argparse.BooleanOptionalAction,
                    help=f"(default: {val})",
                )
            else:
                parser.add_argument(
                    f"--{name}",
                    type=arg_type,
                    default=val,
                    help=f"(default: {val})",
                )

        args = parser.parse_args()
        return cls(**vars(args))
