#!/usr/bin/env python3
"""Aggregate batch inference shard summaries."""

import argparse
from pathlib import Path

from _common import load_yaml, read_json, resolve_path, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    output_root = resolve_path(args.config.parent, str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    raw_dir = output_root / str(cfg["output"]["raw_dir"])
    summary_files = sorted(raw_dir.glob("summary_shard*.json"))
    if not summary_files:
        raise SystemExit(f"No shard summaries found in {raw_dir}")

    summaries = [read_json(path) for path in summary_files]
    total_records = sum(int(row["records_written"]) for row in summaries)
    total_elapsed = sum(float(row["elapsed_seconds"]) for row in summaries)
    elapsed_values = [float(row["elapsed_seconds"]) for row in summaries]
    max_elapsed = max(elapsed_values)
    min_elapsed = min(elapsed_values)
    work_units = [int(row.get("work_units_total", row["records_written"])) for row in summaries]
    shard_rows = [
        {
            "array_index": int(row["array_index"]),
            "records_written": int(row["records_written"]),
            "work_units_total": int(row.get("work_units_total", row["records_written"])),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "throughput_records_per_sec": float(row["throughput_records_per_sec"]),
        }
        for row in summaries
    ]
    aggregate = {
        "run_name": str(cfg["run"]["run_name"]),
        "shards_completed": len(summaries),
        "records_written": total_records,
        "work_units_total": sum(work_units),
        "sum_elapsed_seconds": total_elapsed,
        "min_shard_elapsed_seconds": min_elapsed,
        "max_shard_elapsed_seconds": max_elapsed,
        "shard_elapsed_imbalance_ratio": max_elapsed / max(1e-9, min_elapsed),
        "throughput_records_per_sec_by_max_elapsed": total_records / max(1e-9, max_elapsed),
        "shards": sorted(shard_rows, key=lambda row: row["array_index"]),
        "shard_summary_files": [str(path) for path in summary_files],
    }
    out_path = output_root / str(cfg["output"]["run_summary_json"])
    write_json(out_path, aggregate)
    print(f"BATCH_INFERENCE_SUMMARY={out_path}")
    print(f"RECORDS_WRITTEN={total_records}")


if __name__ == "__main__":
    main()
