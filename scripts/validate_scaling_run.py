#!/usr/bin/env python3
"""Validate synthetic scaling ladder artifacts."""

from pathlib import Path

from _common import load_yaml, read_json, resolve_path


def ensure_file(path):
    if not path.is_file():
        raise SystemExit(f"Missing expected file: {path}")


def main():
    guide_dir = Path(__file__).resolve().parents[1]
    configs_dir = guide_dir / "configs" / "synthetic"
    config_paths = [
        configs_dir / "baseline.yaml",
        configs_dir / "single_node.yaml",
        configs_dir / "two_node.yaml",
    ]

    summaries = {}
    for config_path in config_paths:
        cfg = load_yaml(config_path)
        run_dir = resolve_path(config_path.parent, str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
        raw_dir = run_dir / str(cfg["output"]["raw_dir"])
        summary_path = run_dir / str(cfg["output"]["run_summary_json"])
        environment_path = run_dir / str(cfg["output"].get("environment_json", "environment.json"))
        ensure_file(summary_path)
        ensure_file(environment_path)

        metric_files = sorted(raw_dir.glob(f"{cfg['output']['metrics_prefix']}*.json"))
        placement_files = sorted(raw_dir.glob(f"{cfg['output']['placement_prefix']}*.json"))
        if not metric_files:
            raise SystemExit(f"No metrics files found in {raw_dir}")
        if not placement_files:
            raise SystemExit(f"No placement files found in {raw_dir}")

        summary = read_json(summary_path)
        if not bool(summary.get("world_size_matches_expected", False)):
            raise SystemExit(f"World size mismatch in {summary_path}")
        if not bool(summary.get("node_count_matches_expected", False)):
            raise SystemExit(f"Node count mismatch in {summary_path}")
        if int(summary.get("rank_count", 0)) != len(metric_files):
            raise SystemExit(f"Rank count mismatch in {summary_path}")
        summaries[str(cfg["run"]["run_name"])] = summary

    outputs_dir = resolve_path(configs_dir, str(load_yaml(configs_dir / "baseline.yaml")["run"]["output_dir"]))
    report_json = outputs_dir / "scaling_report.json"
    report_md = outputs_dir / "scaling_report.md"
    ensure_file(report_json)
    ensure_file(report_md)

    print("VALIDATION_OK=1")
    print(f"baseline_world_size={summaries['synthetic-1gcd']['world_size']}")
    print(f"single_node_world_size={summaries['synthetic-8gcd-single-node']['world_size']}")
    print(f"two_node_world_size={summaries['synthetic-16gcd-two-node']['world_size']}")
    print(f"baseline_gpu_visible_count={summaries['synthetic-1gcd']['gpu_visible_count']}")


if __name__ == "__main__":
    main()

