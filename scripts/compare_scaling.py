#!/usr/bin/env python3
"""Print a compact synthetic scaling comparison."""

import argparse
from pathlib import Path

from _common import load_yaml, read_json, resolve_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        help="Config files to compare in order. Defaults to the main 1/8/16-GCD ladder.",
    )
    return parser.parse_args()


def load_summary(config_path):
    cfg = load_yaml(config_path)
    summary_path = (
        resolve_path(config_path.parent, str(cfg["run"]["output_dir"]))
        / str(cfg["run"]["run_name"])
        / str(cfg["output"]["run_summary_json"])
    )
    return cfg, summary_path


def speedup(base, target):
    return target / max(1e-9, base)


def efficiency(speedup_value, base_world, target_world):
    return speedup_value / max(1e-9, target_world / max(1, base_world))


def diagnose(eff):
    if eff >= 0.8:
        return "good scaling efficiency"
    if eff >= 0.5:
        return "moderate scaling efficiency; inspect communication and workload size"
    return "poor scaling efficiency; likely communication or workload-size bottleneck"


def metric_row(label, summary, base_thr, base_world):
    throughput = float(summary["total_throughput_samples_per_sec"])
    world_size = int(summary["world_size"])
    speedup_value = speedup(base_thr, throughput)
    efficiency_value = efficiency(speedup_value, base_world, world_size)
    return {
        "label": label,
        "world_size": world_size,
        "node_count": int(summary.get("node_count", 0) or 0),
        "total_throughput_samples_per_sec": throughput,
        "speedup_vs_baseline": speedup_value,
        "efficiency_vs_baseline": efficiency_value,
        "diagnosis": diagnose(efficiency_value),
    }


def print_table(rows):
    print("| Configuration | World Size | Nodes | Throughput | Speedup | Efficiency | Diagnosis |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['label']} | {row['world_size']} | {row['node_count']} | "
            f"{row['total_throughput_samples_per_sec']:.2f} | "
            f"{row['speedup_vs_baseline']:.2f} | "
            f"{row['efficiency_vs_baseline']:.2f} | {row['diagnosis']} |"
        )


def main():
    args = parse_args()
    guide_dir = Path(__file__).resolve().parents[1]
    configs_dir = guide_dir / "configs" / "synthetic"

    explicit_configs = bool(args.configs)
    config_paths = args.configs or [
        configs_dir / "baseline.yaml",
        configs_dir / "single_node.yaml",
        configs_dir / "two_node.yaml",
    ]

    loaded = []
    for config_path in config_paths:
        cfg, summary_path = load_summary(config_path)
        label = str(cfg["run"]["run_name"]).replace("synthetic-", "")
        if summary_path.is_file():
            loaded.append((label, summary_path, read_json(summary_path)))
        elif explicit_configs:
            raise SystemExit(f"Missing run summary: {summary_path}")

    if not loaded:
        raise SystemExit("No completed run summaries found for the requested configs.")

    base_thr = float(loaded[0][2]["total_throughput_samples_per_sec"])
    base_world = int(loaded[0][2]["world_size"])

    rows = [metric_row(label, summary, base_thr, base_world) for label, _, summary in loaded]
    print_table(rows)

    if len(rows) >= 2:
        previous = rows[-2]
        current = rows[-1]
        incremental_speedup = speedup(
            previous["total_throughput_samples_per_sec"],
            current["total_throughput_samples_per_sec"],
        )
        world_ratio = current["world_size"] / max(1, previous["world_size"])
        incremental_efficiency = incremental_speedup / max(1e-9, world_ratio)
        print("")
        print(f"INCREMENTAL_SPEEDUP={incremental_speedup:.4f}")
        print(f"INCREMENTAL_EFFICIENCY={incremental_efficiency:.4f}")


if __name__ == "__main__":
    main()
