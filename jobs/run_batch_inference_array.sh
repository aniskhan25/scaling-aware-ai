#!/bin/bash

#SBATCH --job-name=sai-batch-inf
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm-%x-%A_%a.out

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export CONTAINER=/appl/local/laifs/containers/lumi-multitorch-latest.sif
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/run_batch_inference.py --config configs/batch-inference/job_array.yaml
"
