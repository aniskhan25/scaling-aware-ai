#!/usr/bin/env python3
"""Print a compact synthetic scaling comparison."""

from pathlib import Path

from _common import load_yaml, read_json, resolve_path


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
        return "moderate scaling efficiency; inspect communication and placement"
    return "poor scaling efficiency; likely communication, placement, or workload-size bottleneck"


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
    guide_dir = Path(__file__).resolve().parents[1]
    configs_dir = guide_dir / "configs" / "synthetic"

    candidates = [
        ("1gcd", configs_dir / "baseline.yaml"),
        ("8gcd-single-node", configs_dir / "single_node.yaml"),
        ("16gcd-two-node", configs_dir / "two_node.yaml"),
    ]

    loaded = []
    for label, config_path in candidates:
        _, summary_path = load_summary(config_path)
        if summary_path.is_file():
            loaded.append((label, summary_path, read_json(summary_path)))

    if not loaded or loaded[0][0] != "1gcd":
        raise SystemExit("Missing baseline summary. Run jobs/run_1gcd.sh first.")

    base_thr = float(loaded[0][2]["total_throughput_samples_per_sec"])
    base_world = int(loaded[0][2]["world_size"])

    rows = [metric_row(label, summary, base_thr, base_world) for label, _, summary in loaded]
    print_table(rows)

    if len(rows) >= 3:
        single = rows[1]["total_throughput_samples_per_sec"]
        two_node = rows[2]["total_throughput_samples_per_sec"]
        incremental_speedup = speedup(single, two_node)
        incremental_efficiency = incremental_speedup / 2.0
        print("")
        print(f"INCREMENTAL_SPEEDUP_8_TO_16={incremental_speedup:.4f}")
        print(f"INCREMENTAL_EFFICIENCY_8_TO_16={incremental_efficiency:.4f}")


if __name__ == "__main__":
    main()
