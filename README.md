# PyTorch Multi-Node Distributed Training (DDP)

Production-grade multi-node, multi-GPU training pipeline built on **PyTorch DistributedDataParallel**.  
Designed for research labs, production ML systems, and technical portfolios.

---

## Architecture

```
┌──────────── Node 0 ────────────┐    ┌──────────── Node 1 ────────────┐
│  GPU 0 ─ Rank 0 (main)        │    │  GPU 0 ─ Rank 4               │
│  GPU 1 ─ Rank 1               │    │  GPU 1 ─ Rank 5               │
│  GPU 2 ─ Rank 2               │◄──►│  GPU 2 ─ Rank 6               │
│  GPU 3 ─ Rank 3               │NCCL│  GPU 3 ─ Rank 7               │
└────────────────────────────────┘    └────────────────────────────────┘
         ▲                                      ▲
         └──────── torchrun orchestrates ───────┘
```

Each rank runs an identical copy of the model.  Gradients are
**all-reduced** across ranks every backward pass; DDP handles this
transparently.  Only rank 0 writes checkpoints and INFO-level logs.

---

## Project Structure

```
├── train.py            # Entry point — training loop
├── config.py           # Dataclass-based configuration + argparse
├── ddp_utils.py        # Distributed setup, teardown, checkpointing
├── model.py            # MiniResNet (lightweight ResNet variant)
├── dataset.py          # Synthetic dataset + DistributedSampler loader
├── utils/
│   ├── logger.py       # Rank-aware structured logging
│   └── metrics.py      # Running averages + distributed all-reduce
├── scripts/
│   └── launch.sh       # Multi-node torchrun wrapper
└── requirements.txt
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt   # torch >= 2.0
```

### Single-node (1–8 GPUs)

```bash
torchrun --standalone --nproc_per_node=2 train.py --epochs 5 --batch_size 64
```

### Multi-node (2 nodes, 4 GPUs each)

**Node 0:**

```bash
MASTER_ADDR=10.0.0.1 NODE_RANK=0 NNODES=2 NPROC_PER_NODE=4 \
    bash scripts/launch.sh --epochs 10
```

**Node 1:**

```bash
MASTER_ADDR=10.0.0.1 NODE_RANK=1 NNODES=2 NPROC_PER_NODE=4 \
    bash scripts/launch.sh --epochs 10
```

### Resume from checkpoint

```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --resume_from checkpoints/checkpoint_epoch_5.pt
```

---

## Configuration Reference

Every parameter is exposed as a CLI flag (see `python train.py --help`).

| Flag | Default | Description |
|------|---------|-------------|
| `--num_classes` | 10 | Number of output classes |
| `--image_size` | 32 | Height/width of input images |
| `--dataset_size` | 50000 | Samples in synthetic dataset |
| `--batch_size` | 64 | Per-GPU batch size |
| `--num_workers` | 4 | DataLoader worker processes |
| `--epochs` | 10 | Total training epochs |
| `--lr` | 0.01 | Initial learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight_decay` | 1e-4 | L2 regularisation |
| `--backend` | nccl | `dist.init_process_group` backend |
| `--use_amp` / `--no-use_amp` | True | Automatic mixed precision |
| `--grad_accum_steps` | 1 | Gradient accumulation micro-steps |
| `--checkpoint_dir` | ./checkpoints | Checkpoint save directory |
| `--save_every` | 1 | Save checkpoint every N epochs |
| `--resume_from` | None | Path to checkpoint to resume from |
| `--log_interval` | 10 | Log every N training steps |
| `--enable_profiling` | False | Enable PyTorch profiler |
| `--seed` | 42 | Random seed (offset by rank) |

---

## Features

- **DistributedDataParallel** — gradient all-reduce via NCCL, no manual sync code.
- **Mixed Precision (AMP)** — `torch.autocast` + `GradScaler` for faster training and lower memory.
- **Gradient Accumulation** — simulate larger effective batch sizes across micro-steps.
- **Cosine Annealing LR** — smooth learning-rate decay over the full training run.
- **Rank-aware Logging** — only rank 0 prints; structured format with timestamps.
- **Checkpoint Resume** — saves and restores model, optimizer, scaler, and epoch state.
- **PyTorch Profiler Integration** — optional CPU/CUDA profiling with TensorBoard export.
- **Synthetic Dataset** — zero external dependencies; drop in any real dataset.

---

## How DDP Works (Brief)

1. **torchrun** spawns one process per GPU across all nodes.
2. Each process calls `dist.init_process_group(backend="nccl")`.
3. The model is replicated and wrapped with `DistributedDataParallel`.
4. A `DistributedSampler` partitions the dataset so each rank sees a unique shard.
5. During `backward()`, DDP hooks **all-reduce** gradients across ranks.
6. Every rank applies the same optimizer step, keeping parameters in sync.

---

## Extending the Pipeline

**Use a real dataset** — replace `SyntheticImageDataset` in `dataset.py` with your own `Dataset` subclass.  
**Swap the model** — edit `create_model()` in `model.py` to return any `nn.Module`.  
**Add validation** — duplicate the epoch loop with `model.eval()` and `torch.no_grad()`.  
**Scale further** — adjust `NNODES` and `NPROC_PER_NODE` in `launch.sh`; the code handles the rest.

---

## License

MIT
