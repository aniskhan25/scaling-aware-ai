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
    placement_files = list_json_files(raw_dir, str(cfg["output"]["placement_prefix"]))
    if not metric_files:
        raise SystemExit(f"No metrics files found in {raw_dir}")

    metrics = [read_json(path) for path in metric_files]
    placements = [read_json(path) for path in placement_files] if placement_files else []

    world_sizes = {int(row["world_size"]) for row in metrics}
    if len(world_sizes) != 1:
        raise SystemExit(f"Inconsistent world sizes in metrics files: {world_sizes}")
    world_size = world_sizes.pop()

    expected_world_size = int(cfg["distributed"]["expected_world_size"])
    expected_nodes = int(cfg["distributed"]["expected_nodes"])
    hostnames = sorted({row.get("hostname", "") for row in placements if row.get("hostname", "")})
    node_count = len(hostnames) if hostnames else 0

    throughputs = [float(row["throughput_samples_per_sec"]) for row in metrics]
    elapsed_seconds = [float(row["elapsed_seconds"]) for row in metrics]
    samples_per_step = int(metrics[0]["samples_per_step"])
    gpu_visible_count = int(metrics[0].get("gpu_visible_count", 0))

    summary = {
        "run_name": run_dir.name,
        "world_size": world_size,
        "expected_world_size": expected_world_size,
        "world_size_matches_expected": world_size == expected_world_size,
        "node_count": node_count,
        "expected_nodes": expected_nodes,
        "node_count_matches_expected": node_count == expected_nodes,
        "gpu_visible_count": gpu_visible_count,
        "rank_count": len(metrics),
        "effective_samples_per_step": samples_per_step * world_size,
        "mean_rank_throughput_samples_per_sec": statistics.mean(throughputs),
        "min_rank_throughput_samples_per_sec": min(throughputs),
        "max_rank_throughput_samples_per_sec": max(throughputs),
        "total_throughput_samples_per_sec": sum(throughputs),
        "max_elapsed_seconds": max(elapsed_seconds),
        "min_elapsed_seconds": min(elapsed_seconds),
        "rank_elapsed_spread_seconds": max(elapsed_seconds) - min(elapsed_seconds),
        "hostnames": hostnames,
        "raw_metrics_files": [str(path) for path in metric_files],
        "raw_placement_files": [str(path) for path in placement_files],
    }

    out_path = run_dir / str(cfg["output"]["run_summary_json"])
    write_json(out_path, summary)

    print(f"RUN_SUMMARY={out_path}")
    print(f"WORLD_SIZE={world_size}")
    print(f"TOTAL_THROUGHPUT={summary['total_throughput_samples_per_sec']:.4f}")


if __name__ == "__main__":
    main()

