# 2. LUMI-G Mental Model

For AI users, the most important thing to understand is the software-visible shape of a LUMI-G node.

A full LUMI-G node exposes:

- 4 AMD MI250X modules
- 2 GCDs per MI250X
- 8 GPU-visible devices per node
- 56 CPU cores available to jobs
- 4 CPU NUMA domains
- Slingshot network connectivity for multi-node communication

In PyTorch, the MI250X GCDs appear through the CUDA-compatible API even though the system uses AMD GPUs and ROCm/HIP underneath. This is why scripts commonly call `torch.cuda.device_count()` on LUMI-G.

## Why This Matters

The distinction between MI250X module and software-visible GCD matters because a "1 GPU" job in framework terms usually means one visible GCD, not one full MI250X module. A full LUMI-G node is therefore an 8-device node from the point of view of PyTorch distributed jobs.

That affects:

- `--gpus-per-node`
- `--nproc_per_node`
- rank-to-device mapping
- global batch size
- per-rank memory
- all-reduce behavior
- scaling efficiency calculations

## Communication Layers

Scaling from 1 GCD to 8 GCDs mainly introduces intra-node communication.

Scaling from 8 GCDs to 16 GCDs introduces inter-node communication as well.

This gives the basic ladder its diagnostic value:

```text
1 GCD                 baseline compute and local overhead
8 GCDs, 1 node        intra-node scaling
16 GCDs, 2 nodes      intra-node plus inter-node scaling
```

If 8-GCD scaling is already poor, fix the local workload, launch, placement, or data path before interpreting multi-node behavior. If 8-GCD scaling is reasonable but 16-GCD scaling drops sharply, inter-node communication or synchronization is more likely to be the limiting factor.

## Placement Metadata To Record

Every rank should record:

- global rank
- local rank
- world size
- hostname
- visible GPU count
- selected device
- CPU affinity
- `SLURM_PROCID`
- `SLURM_LOCALID`
- `SLURM_NODEID`
- `ROCR_VISIBLE_DEVICES`
- `CUDA_VISIBLE_DEVICES`

Without this metadata, a poor scaling result is difficult to interpret. You may be measuring a launch mistake instead of a workload property.

## Practical Rule

Validate placement before interpreting performance.

