#!/usr/bin/env python3
"""Run a shardable batch inference workload for job-array examples."""

import argparse
import json
import time
from pathlib import Path

from _common import load_yaml, resolve_path, write_json

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is required. Run inside the LUMI AI container.") from exc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--array-count", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def shard_rows(rows, array_index, array_count):
    return [row for idx, row in enumerate(rows) if idx % array_count == array_index]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    config_dir = args.config.parent

    array_index = args.array_index
    if array_index is None:
        import os

        array_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    array_count = args.array_count
    if array_count is None:
        import os

        array_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        array_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", "0"))
        array_count = max(1, array_max - array_min + 1)

    if not torch.cuda.is_available() and not bool(cfg["runtime"]["allow_cpu_fallback"]):
        raise SystemExit("GPU required but no CUDA/HIP device is visible through PyTorch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_visible_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    input_path = resolve_path(config_dir, str(cfg["data"]["input_jsonl"]))
    output_root = resolve_path(config_dir, str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    raw_dir = output_root / str(cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    max_records = int(cfg["data"]["max_records"])
    if max_records > 0:
        rows = rows[:max_records]
    shard = shard_rows(rows, array_index, array_count)

    batch_size = int(cfg["inference"]["batch_size"])
    input_dim = int(cfg["inference"]["input_dim"])
    compute_repeats = int(cfg["inference"]["compute_repeats"])
    work_units_key = str(cfg["data"].get("work_units_key", ""))
    work_unit_sleep_seconds = float(cfg["inference"].get("work_unit_sleep_seconds", 0.0))
    text_key = str(cfg["data"]["text_key"])
    id_key = str(cfg["data"]["id_key"])

    torch.manual_seed(int(cfg["run"]["seed"]) + array_index)
    weight = torch.randn(input_dim, input_dim, device=device)
    output_jsonl = raw_dir / f"outputs_shard{array_index}.jsonl"

    start = time.perf_counter()
    total = 0
    work_units_total = 0
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for batch_rows in batched(shard, batch_size):
            batch_work_units = 0
            if work_units_key:
                batch_work_units = sum(int(row.get(work_units_key, 1)) for row in batch_rows)
                work_units_total += batch_work_units
                if work_unit_sleep_seconds > 0:
                    time.sleep(batch_work_units * work_unit_sleep_seconds)
            x = torch.randn(len(batch_rows), input_dim, device=device)
            y = x
            for _ in range(compute_repeats):
                y = torch.relu(y @ weight)
            sync(device)
            scores = y.mean(dim=1).detach().cpu().tolist()
            for row, score in zip(batch_rows, scores, strict=True):
                out_f.write(
                    json.dumps(
                        {
                            "id": row[id_key],
                            "input_chars": len(str(row[text_key])),
                            "score": float(score),
                            "array_index": array_index,
                            "work_units": int(row.get(work_units_key, 1)) if work_units_key else 1,
                        }
                    )
                    + "\n"
                )
                total += 1

    sync(device)
    elapsed = max(1e-9, time.perf_counter() - start)
    summary = {
        "mode": "batch-inference-shard",
        "run_name": str(cfg["run"]["run_name"]),
        "array_index": array_index,
        "array_count": array_count,
        "device": str(device),
        "gpu_visible_count": gpu_visible_count,
        "records_available": len(rows),
        "records_in_shard": len(shard),
        "records_written": total,
        "work_units_total": work_units_total if work_units_key else total,
        "work_units_key": work_units_key,
        "work_unit_sleep_seconds": work_unit_sleep_seconds,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "throughput_records_per_sec": total / elapsed,
        "output_jsonl": str(output_jsonl),
    }
    summary_path = raw_dir / f"summary_shard{array_index}.json"
    write_json(summary_path, summary)

    print("RUN_COMPLETE=1")
    print(f"SHARD_SUMMARY={summary_path}")
    print(f"RECORDS_WRITTEN={total}")


if __name__ == "__main__":
    main()
