#!/usr/bin/env python3
"""Build a hands-on LUMI lab report from completed run artifacts."""

from pathlib import Path

from _common import load_yaml, read_json, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def summary_path(config_path):
    cfg = load_yaml(config_path)
    run_dir = resolve_path(config_path.parent, str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    return run_dir / str(cfg["output"]["run_summary_json"])


def optional_json(path):
    if not path.is_file():
        return None
    return read_json(path)


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def status(ok):
    return "complete" if ok else "missing"


def add_synthetic_section(lines):
    report = optional_json(OUTPUTS_DIR / "scaling_report.json")
    lines.extend(["## Lab 1: Scaling Ladder", ""])
    if not report:
        lines.extend(
            [
                "Status: missing",
                "",
                "Run these jobs first:",
                "",
                "```bash",
                "sbatch jobs/run_1gcd.sh",
                "sbatch jobs/run_8gcd_single_node.sh",
                "sbatch jobs/run_16gcd_two_node.sh",
                "python scripts/compare_scaling.py",
                "python scripts/validate_scaling_run.py",
                "```",
                "",
            ]
        )
        return

    lines.extend(
        [
            "Status: complete",
            "",
            "| Run | World size | Nodes | Throughput | Speedup | Efficiency | Diagnosis |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {row['world_size']} | {row['node_count']} | "
            f"{fmt(row['total_throughput_samples_per_sec'])} | "
            f"{fmt(row['speedup_vs_baseline'])} | "
            f"{fmt(row['efficiency_vs_baseline'])} | {row['diagnosis']} |"
        )
    lines.extend(
        [
            "",
            "Decision prompt: move up only where efficiency and placement make the larger run defensible.",
            "",
        ]
    )


def add_ddp_bottleneck_section(lines):
    bottleneck = optional_json(
        summary_path(REPO_ROOT / "configs" / "bottlenecks" / "ddp_data_wait_bottleneck.yaml")
    )
    fixed = optional_json(
        summary_path(REPO_ROOT / "configs" / "bottlenecks" / "ddp_data_wait_reduced.yaml")
    )
    lines.extend(["## Lab 2: DDP Data Starvation", ""])
    lines.append(f"Status: bottleneck={status(bottleneck is not None)}, fixed={status(fixed is not None)}")
    lines.append("")
    if not (bottleneck and fixed):
        lines.extend(
            [
                "Run the induced bottleneck and the resolution:",
                "",
                "```bash",
                "sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_bottleneck.yaml jobs/run_ddp_8gcd_config.sh",
                "sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_reduced.yaml jobs/run_ddp_8gcd_config.sh",
                "```",
                "",
            ]
        )
        return

    lines.extend(
        [
            "| Case | Throughput | Mean data wait | Max data wait | Rank elapsed spread |",
            "|---|---:|---:|---:|---:|",
            f"| Induced bottleneck | {fmt(bottleneck['total_throughput_samples_per_sec'])} | "
            f"{fmt(100 * bottleneck.get('mean_data_wait_fraction', 0.0), 1)}% | "
            f"{fmt(100 * bottleneck.get('max_data_wait_fraction', 0.0), 1)}% | "
            f"{fmt(bottleneck['rank_elapsed_spread_seconds'])} |",
            f"| Reduced data wait | {fmt(fixed['total_throughput_samples_per_sec'])} | "
            f"{fmt(100 * fixed.get('mean_data_wait_fraction', 0.0), 1)}% | "
            f"{fmt(100 * fixed.get('max_data_wait_fraction', 0.0), 1)}% | "
            f"{fmt(fixed['rank_elapsed_spread_seconds'])} |",
            "",
            "Resolution prompt: if the fixed case improves throughput by reducing wait, the next real fix is the input pipeline, not more nodes.",
            "",
        ]
    )


def add_job_array_section(lines):
    imbalanced = optional_json(
        summary_path(REPO_ROOT / "configs" / "bottlenecks" / "job_array_imbalanced.yaml")
    )
    balanced = optional_json(
        summary_path(REPO_ROOT / "configs" / "bottlenecks" / "job_array_balanced.yaml")
    )
    lines.extend(["## Lab 3: Job-Array Shard Imbalance", ""])
    lines.append(f"Status: imbalanced={status(imbalanced is not None)}, balanced={status(balanced is not None)}")
    lines.append("")
    if not (imbalanced and balanced):
        lines.extend(
            [
                "Run the imbalanced shards and the balanced resolution:",
                "",
                "```bash",
                "sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_imbalanced.yaml jobs/run_batch_inference_array_config.sh",
                "python scripts/collect_batch_inference.py --config configs/bottlenecks/job_array_imbalanced.yaml",
                "sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_balanced.yaml jobs/run_batch_inference_array_config.sh",
                "python scripts/collect_batch_inference.py --config configs/bottlenecks/job_array_balanced.yaml",
                "```",
                "",
            ]
        )
        return

    lines.extend(
        [
            "| Case | Shards | Records | Max shard elapsed | Imbalance ratio | Throughput by slowest shard |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Imbalanced | {imbalanced['shards_completed']} | {imbalanced['records_written']} | "
            f"{fmt(imbalanced['max_shard_elapsed_seconds'])} | "
            f"{fmt(imbalanced['shard_elapsed_imbalance_ratio'])} | "
            f"{fmt(imbalanced['throughput_records_per_sec_by_max_elapsed'])} |",
            f"| Balanced | {balanced['shards_completed']} | {balanced['records_written']} | "
            f"{fmt(balanced['max_shard_elapsed_seconds'])} | "
            f"{fmt(balanced['shard_elapsed_imbalance_ratio'])} | "
            f"{fmt(balanced['throughput_records_per_sec_by_max_elapsed'])} |",
            "",
            "Resolution prompt: independent records usually want better sharding before distributed collectives.",
            "",
        ]
    )


def main():
    lines = [
        "# Hands-On LUMI Scaling Lab Report",
        "",
        "This report is generated from artifacts under `outputs/`.",
        "Missing sections list the commands needed to produce the evidence.",
        "",
    ]
    add_synthetic_section(lines)
    add_ddp_bottleneck_section(lines)
    add_job_array_section(lines)
    lines.extend(
        [
            "## Final Decision Record",
            "",
            "Copy the relevant measurements into `templates/scale-decision-record.md` and make a stop/go decision before increasing scale.",
            "",
        ]
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "lumi_hands_on_lab_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"LAB_REPORT={out_path}")


if __name__ == "__main__":
    main()

