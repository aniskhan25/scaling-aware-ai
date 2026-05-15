#!/bin/bash

#SBATCH --job-name=sai-16gcd
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export CONTAINER=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="1${SLURM_JOB_ID:0-4}"
export WORLD_SIZE=$SLURM_NPROCS

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST SLURM_NTASKS=$SLURM_NTASKS"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/summarize_environment.py --config configs/synthetic/two_node.yaml
"

srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
python scripts/inspect_placement.py --config configs/synthetic/two_node.yaml
"

srun --cpu-bind=v,mask_cpu=$CPU_BIND_MASKS singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
python scripts/run_synthetic_workload.py --config configs/synthetic/two_node.yaml
"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/collect_metrics.py --config configs/synthetic/two_node.yaml
"
