#!/bin/bash

#SBATCH --job-name=sai-ddp-1gcd
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export CONTAINER=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/summarize_environment.py --config configs/ddp-training/baseline.yaml
python scripts/inspect_placement.py --config configs/ddp-training/baseline.yaml
python scripts/run_ddp_training.py --config configs/ddp-training/baseline.yaml
python scripts/collect_metrics.py --config configs/ddp-training/baseline.yaml
"
