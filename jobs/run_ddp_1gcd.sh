#!/bin/bash

#SBATCH --job-name=sai-ddp-1gcd
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

if [ -f ../env.sh ]; then
  source ../env.sh
fi
: "${CONTAINER:?Set CONTAINER to a valid LUMI AI container path.}"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/summarize_environment.py --config configs/ddp-training/baseline.yaml
python scripts/inspect_placement.py --config configs/ddp-training/baseline.yaml
python scripts/run_ddp_training.py --config configs/ddp-training/baseline.yaml
python scripts/collect_metrics.py --config configs/ddp-training/baseline.yaml
"

