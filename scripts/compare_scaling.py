#!/usr/bin/env python3
"""Compare synthetic scaling summaries and build a compact report."""

from pathlib import Path

from _common import load_yaml, read_json, resolve_path, write_json


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


def main():
    guide_dir = Path(__file__).resolve().parents[1]
    configs_dir = guide_dir / "configs" / "synthetic"

    baseline_cfg, baseline_summary_path = load_summary(configs_dir / "baseline.yaml")
    _, single_summary_path = load_summary(configs_dir / "single_node.yaml")
    _, two_node_summary_path = load_summary(configs_dir / "two_node.yaml")

    baseline_summary = read_json(baseline_summary_path)
    single_summary = read_json(single_summary_path)
    two_node_summary = read_json(two_node_summary_path)

    base_thr = float(baseline_summary["total_throughput_samples_per_sec"])
    base_world = int(baseline_summary["world_size"])

    rows = [
        metric_row("1gcd", baseline_summary, base_thr, base_world),
        metric_row("8gcd-single-node", single_summary, base_thr, base_world),
        metric_row("16gcd-two-node", two_node_summary, base_thr, base_world),
    ]

    outputs_dir = resolve_path((configs_dir / "baseline.yaml").parent, str(baseline_cfg["run"]["output_dir"]))
    out_json = outputs_dir / "scaling_report.json"
    out_md = outputs_dir / "scaling_report.md"

    payload = {
        "baseline_summary_path": str(baseline_summary_path),
        "single_node_summary_path": str(single_summary_path),
        "two_node_summary_path": str(two_node_summary_path),
        "rows": rows,
    }
    write_json(out_json, payload)

    lines = [
        "# Synthetic Scaling Report",
        "",
        "| Configuration | World Size | Nodes | Throughput | Speedup | Efficiency | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['world_size']} | {row['node_count']} | "
            f"{row['total_throughput_samples_per_sec']:.2f} | {row['speedup_vs_baseline']:.2f} | "
            f"{row['efficiency_vs_baseline']:.2f} | {row['diagnosis']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prefer a larger configuration only if useful throughput improves enough to justify the added communication and GPU-hours.",
            "- Validate placement metadata before interpreting poor scaling as a model or framework problem.",
            "- If single-node scaling is already poor, fix that before trusting multi-node results.",
        ]
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"SCALING_REPORT_JSON={out_json}")
    print(f"SCALING_REPORT_MD={out_md}")


if __name__ == "__main__":
    main()

