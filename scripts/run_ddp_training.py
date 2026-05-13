#!/usr/bin/env python3
"""Run a compact synthetic DDP training workload with workload-specific metrics."""

import argparse
import time
from pathlib import Path

from _common import load_yaml, rank_info, resolve_run_dir, write_json

try:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
except ImportError as exc:
    raise SystemExit("PyTorch is required. Run inside the LUMI AI container.") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


class TinyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, depth):
        super().__init__()
        layers = []
        current = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(current, hidden_dim), nn.GELU()])
            current = hidden_dim
        layers.append(nn.Linear(current, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_distributed(world_size):
    if world_size <= 1:
        return False
    if not dist.is_available():
        raise SystemExit("torch.distributed is not available.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return True


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def synthetic_batch(batch_size, input_dim, num_classes, device):
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)
    raw_dir = run_dir / str(cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    info = rank_info()
    rank = int(info["rank"])
    local_rank = int(info["local_rank"])
    world_size = int(info["world_size"])

    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("GPU required but no CUDA/HIP device is visible through PyTorch.")

    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        device_index = local_rank % max(1, gpu_visible_count)
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    is_distributed = init_distributed(world_size)

    seed = int(cfg["run"]["seed"]) + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    input_dim = int(cfg["model"]["input_dim"])
    hidden_dim = int(cfg["model"]["hidden_dim"])
    num_classes = int(cfg["model"]["num_classes"])
    depth = int(cfg["model"]["depth"])
    model = TinyClassifier(input_dim, hidden_dim, num_classes, depth).to(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device.index])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    local_batch_size = int(cfg["training"]["local_batch_size"])
    steps = int(cfg["training"]["steps"])
    warmup_steps = int(cfg["training"]["warmup_steps"])
    data_wait_seconds = float(cfg["training"].get("synthetic_data_wait_seconds", 0.0))

    def train_step():
        wait_start = time.perf_counter()
        if data_wait_seconds > 0:
            time.sleep(data_wait_seconds)
        x, y = synthetic_batch(local_batch_size, input_dim, num_classes, device)
        data_wait = time.perf_counter() - wait_start

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        sync(device)
        return float(loss.detach().cpu()), data_wait

    for _ in range(warmup_steps):
        train_step()

    if is_distributed:
        dist.barrier()
    sync(device)

    losses = []
    data_wait_total = 0.0
    start = time.perf_counter()
    for _ in range(steps):
        loss_value, data_wait = train_step()
        losses.append(loss_value)
        data_wait_total += data_wait
    if is_distributed:
        dist.barrier()
    sync(device)
    elapsed = max(1e-9, time.perf_counter() - start)

    checkpoint_seconds = 0.0
    checkpoint_path = ""
    if bool(cfg["training"]["write_checkpoint"]) and rank == 0:
        checkpoint_start = time.perf_counter()
        checkpoint_path = str(run_dir / "checkpoint.pt")
        state_dict = model.module.state_dict() if is_distributed else model.state_dict()
        torch.save({"model": state_dict, "config": cfg}, checkpoint_path)
        checkpoint_seconds = time.perf_counter() - checkpoint_start

    local_samples = steps * local_batch_size
    local_throughput = local_samples / elapsed
    global_samples = local_samples * world_size
    global_throughput = global_samples / elapsed

    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "samples_per_step": local_batch_size,
        "local_batch_size": local_batch_size,
        "global_batch_size": local_batch_size * world_size,
        "elapsed_seconds": elapsed,
        "local_samples": local_samples,
        "global_samples_if_all_ranks": global_samples,
        "throughput_samples_per_sec": local_throughput,
        "global_throughput_samples_per_sec_if_same_elapsed": global_throughput,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "data_wait_seconds_total": data_wait_total,
        "data_wait_fraction": data_wait_total / elapsed,
        "checkpoint_seconds_rank0": checkpoint_seconds,
        "checkpoint_path_rank0": checkpoint_path,
    }

    out_path = raw_dir / f"{cfg['output']['metrics_prefix']}{rank}.json"
    write_json(out_path, payload)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    print(f"TRAINING_METRICS_PATH={out_path}")
    print(f"RANK={rank}")
    print(f"LOCAL_THROUGHPUT={local_throughput:.4f}")


if __name__ == "__main__":
    main()
