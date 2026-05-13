#!/bin/bash

#SBATCH --job-name=sai-16gcd
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:40:00

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

if [ -f ../env.sh ]; then
  source ../env.sh
fi
: "${CONTAINER:?Set CONTAINER to a valid LUMI AI container path.}"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"

srun --cpu-bind=cores --distribution=block:block singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
if [ \"\${SLURM_PROCID:-0}\" = \"0\" ]; then
  python scripts/summarize_environment.py --config configs/synthetic/two_node.yaml
fi
python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint='$MASTER_ADDR:$MASTER_PORT' scripts/inspect_placement.py --config configs/synthetic/two_node.yaml
python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint='$MASTER_ADDR:$MASTER_PORT' scripts/run_synthetic_workload.py --config configs/synthetic/two_node.yaml
if [ \"\${SLURM_PROCID:-0}\" = \"0\" ]; then
  python scripts/collect_metrics.py --config configs/synthetic/two_node.yaml
fi
"
