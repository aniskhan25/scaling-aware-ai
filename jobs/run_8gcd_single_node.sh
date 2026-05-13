#!/bin/bash

#SBATCH --job-name=sai-8gcd
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:30:00

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
python scripts/summarize_environment.py --config configs/synthetic/single_node.yaml
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/inspect_placement.py --config configs/synthetic/single_node.yaml
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/run_synthetic_workload.py --config configs/synthetic/single_node.yaml
python scripts/collect_metrics.py --config configs/synthetic/single_node.yaml
"

