#!/usr/bin/env python3
"""Capture per-rank placement metadata."""

import argparse
import os
import socket
from pathlib import Path

from _common import load_yaml, rank_info, resolve_run_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def detect_gpu():
    info = {"gpu_visible_count": 0, "torch_available": False}
    try:
        import torch

        info["torch_available"] = True
        info["gpu_visible_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["current_device"] = int(torch.cuda.current_device())
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    return info


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)
    raw_dir = run_dir / str(cfg["output"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    info = rank_info()
    rank = int(info["rank"])
    payload = {
        **info,
        **detect_gpu(),
        "hostname": socket.gethostname(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_localid": os.environ.get("SLURM_LOCALID", ""),
        "slurm_procid": os.environ.get("SLURM_PROCID", ""),
        "slurm_nodeid": os.environ.get("SLURM_NODEID", ""),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }

    out_path = raw_dir / f"{cfg['output']['placement_prefix']}{rank}.json"
    write_json(out_path, payload)

    print(f"PLACEMENT_PATH={out_path}")
    print(f"RANK={rank}")


if __name__ == "__main__":
    main()

