#!/usr/bin/env python3
"""Aggregate per-rank metrics into one run summary."""

import argparse
import statistics
from pathlib import Path

from _common import list_json_files, load_yaml, read_json, resolve_run_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)
    raw_dir = run_dir / str(cfg["output"]["raw_dir"])

    metric_files = list_json_files(raw_dir, str(cfg["output"]["metrics_prefix"]))
    if not metric_files:
        raise SystemExit(f"No metrics files found in {raw_dir}")

    metrics = [read_json(path) for path in metric_files]

    world_sizes = {int(row["world_size"]) for row in metrics}
    if len(world_sizes) != 1:
        raise SystemExit(f"Inconsistent world sizes in metrics files: {world_sizes}")
    world_size = world_sizes.pop()

    hostnames = sorted({row.get("hostname", "") for row in metrics if row.get("hostname", "")})
    node_count = len(hostnames) if hostnames else 0

    throughputs = [float(row["throughput_samples_per_sec"]) for row in metrics]
    elapsed_seconds = [float(row["elapsed_seconds"]) for row in metrics]
    max_elapsed = max(elapsed_seconds)
    min_elapsed = min(elapsed_seconds)
    samples_per_step = int(metrics[0]["samples_per_step"])
    local_samples = [
        int(row.get("local_samples", int(row["steps"]) * int(row["samples_per_step"])))
        for row in metrics
    ]
    total_samples = sum(local_samples)
    gpu_visible_count = int(metrics[0].get("gpu_visible_count", 0))
    data_wait_fractions = [
        float(row["data_wait_fraction"]) for row in metrics if "data_wait_fraction" in row
    ]

    summary = {
        "run_name": run_dir.name,
        "world_size": world_size,
        "node_count": node_count,
        "gpu_visible_count": gpu_visible_count,
        "rank_count": len(metrics),
        "effective_samples_per_step": samples_per_step * world_size,
        "mean_rank_throughput_samples_per_sec": statistics.mean(throughputs),
        "min_rank_throughput_samples_per_sec": min(throughputs),
        "max_rank_throughput_samples_per_sec": max(throughputs),
        "sum_rank_throughput_samples_per_sec": sum(throughputs),
        "total_samples": total_samples,
        "total_throughput_samples_per_sec": total_samples / max(1e-9, max_elapsed),
        "max_elapsed_seconds": max_elapsed,
        "min_elapsed_seconds": min_elapsed,
        "rank_elapsed_spread_seconds": max_elapsed - min_elapsed,
        "hostnames": hostnames,
        "raw_metrics_files": [str(path) for path in metric_files],
    }
    if data_wait_fractions:
        summary.update(
            {
                "mean_data_wait_fraction": statistics.mean(data_wait_fractions),
                "max_data_wait_fraction": max(data_wait_fractions),
            }
        )
    out_path = run_dir / str(cfg["output"]["run_summary_json"])
    write_json(out_path, summary)

    print(f"RUN_SUMMARY={out_path}")
    print(f"WORLD_SIZE={world_size}")
    print(f"NODES={summary['node_count']}")
    print(f"RANKS={summary['rank_count']}")
    print(f"TOTAL_THROUGHPUT={summary['total_throughput_samples_per_sec']:.4f}")
    print(f"RANK_ELAPSED_SPREAD={summary['rank_elapsed_spread_seconds']:.4f}")
    if data_wait_fractions:
        print(f"MEAN_DATA_WAIT_FRACTION={summary['mean_data_wait_fraction']:.4f}")
        print(f"MAX_DATA_WAIT_FRACTION={summary['max_data_wait_fraction']:.4f}")


if __name__ == "__main__":
    main()
