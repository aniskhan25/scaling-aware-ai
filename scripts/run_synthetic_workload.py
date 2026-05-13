#!/usr/bin/env python3
"""Run a compact synthetic scaling workload and emit per-rank metrics."""

import argparse
import time
from pathlib import Path

from _common import load_yaml, rank_info, resolve_run_dir, write_json

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is required. Run inside the LUMI AI container.") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def maybe_init_distributed(world_size):
    if world_size <= 1:
        return False
    if not torch.distributed.is_available():
        raise SystemExit("torch.distributed is not available in this environment.")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    return True


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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

    require_gpu = bool(cfg["runtime"]["require_gpu"])
    allow_cpu_fallback = bool(cfg["runtime"]["allow_cpu_fallback"])
    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if require_gpu and gpu_visible_count < 1 and not allow_cpu_fallback:
        raise SystemExit("GPU required but no CUDA/HIP device is visible through PyTorch.")

    if torch.cuda.is_available():
        device_index = local_rank % max(1, gpu_visible_count)
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    is_distributed = maybe_init_distributed(world_size)
    if is_distributed:
        torch.distributed.barrier()

    samples_per_step = int(cfg["workload"]["samples_per_step"])
    steps = int(cfg["workload"]["steps"])
    warmup_steps = int(cfg["workload"]["warmup_steps"])
    hidden_size = int(cfg["workload"]["hidden_size"])
    compute_repeats = int(cfg["workload"]["compute_repeats"])

    x = torch.randn(samples_per_step, hidden_size, device=device)
    w = torch.randn(hidden_size, hidden_size, device=device)
    b = torch.randn(hidden_size, device=device)

    def step_body():
        y = x
        for _ in range(compute_repeats):
            y = torch.relu(torch.matmul(y, w) + b)
        if is_distributed:
            comm_buf = y.mean(dim=0)
            torch.distributed.all_reduce(comm_buf)
        sync(device)

    for _ in range(warmup_steps):
        step_body()

    if is_distributed:
        torch.distributed.barrier()
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        step_body()
    if is_distributed:
        torch.distributed.barrier()
    sync(device)
    elapsed = max(1e-9, time.perf_counter() - start)

    throughput = (max(1, steps) * samples_per_step) / elapsed
    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "samples_per_step": samples_per_step,
        "hidden_size": hidden_size,
        "compute_repeats": compute_repeats,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_sec": throughput,
    }

    out_path = raw_dir / f"{cfg['output']['metrics_prefix']}{rank}.json"
    write_json(out_path, payload)

    if is_distributed:
        torch.distributed.barrier()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    print(f"METRICS_PATH={out_path}")
    print(f"RANK={rank}")
    print(f"THROUGHPUT={throughput:.4f}")


if __name__ == "__main__":
    main()

