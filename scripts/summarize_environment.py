#!/usr/bin/env python3
"""Record lightweight environment metadata for a scaling run."""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from _common import load_yaml, resolve_run_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def run_text(command):
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return (result.stdout or result.stderr).strip()


def torch_info():
    info = {"available": False}
    try:
        import torch

        info.update(
            {
                "available": True,
                "version": getattr(torch, "__version__", ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "hip_version": getattr(torch.version, "hip", None),
                "cuda_version": getattr(torch.version, "cuda", None),
            }
        )
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_dir = resolve_run_dir(cfg, args.config)

    payload = {
        "run_name": cfg["run"]["run_name"],
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch_info(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
            "job_num_nodes": os.environ.get("SLURM_JOB_NUM_NODES", ""),
            "submit_dir": os.environ.get("SLURM_SUBMIT_DIR", ""),
        },
        "environment": {
            "CONTAINER": os.environ.get("CONTAINER", ""),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", ""),
            "NCCL_NET_GDR_LEVEL": os.environ.get("NCCL_NET_GDR_LEVEL", ""),
        },
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
    }

    out_path = run_dir / str(cfg["output"].get("environment_json", "environment.json"))
    write_json(out_path, payload)
    print(f"ENVIRONMENT_SUMMARY={out_path}")


if __name__ == "__main__":
    main()

