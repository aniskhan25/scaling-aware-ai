#!/usr/bin/env python3
"""Shared helpers for Scaling-Aware AI examples."""

import json
import os
from pathlib import Path


def load_yaml(path):
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        return load_simple_yaml(path)


def parse_scalar(value):
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_simple_yaml(path):
    """Parse the small nested mapping subset used by this guide's configs."""
    root = {}
    stack = [(-1, root)]
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            key, sep, value = line.strip().partition(":")
            if not sep:
                raise ValueError(f"Unsupported YAML line in {path}: {raw_line.rstrip()}")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value.strip():
                parent[key] = parse_scalar(value)
            else:
                parent[key] = {}
                stack.append((indent, parent[key]))
    return root


def resolve_path(base_dir, raw_path):
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_run_dir(cfg, cfg_path):
    run_dir = resolve_path(cfg_path.parent, str(cfg["run"]["output_dir"])) / str(cfg["run"]["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def rank_info():
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(path, prefix):
    if not path.is_dir():
        return []
    return sorted(file for file in path.glob(f"{prefix}*.json") if file.is_file())
