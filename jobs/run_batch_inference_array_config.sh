#!/bin/bash

#SBATCH --job-name=sai-batch-config
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=standard-g
#SBATCH --array=0-7
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
: "${CONFIG:?Set CONFIG to a batch inference config path.}"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/run_batch_inference.py --config '$CONFIG'
"

