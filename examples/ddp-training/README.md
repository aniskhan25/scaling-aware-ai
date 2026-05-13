# DDP Training Example

This example shows a workload where synchronized distributed training is appropriate.

The script trains a compact synthetic classifier with PyTorch. It records training throughput, loss, data wait fraction, checkpoint time, rank placement, and per-rank metrics.

The purpose is not model quality. The purpose is to show how training metrics differ from the synthetic scaling ladder:

- local batch size matters
- global batch size changes with world size
- gradients synchronize every step
- checkpointing can affect wall-clock time
- data wait can be simulated and measured

## Run

From the `scaling-aware-ai` directory, edit the account line in the job scripts, then submit:

```bash
sbatch jobs/run_ddp_1gcd.sh
sbatch jobs/run_ddp_8gcd_single_node.sh
```

After both jobs finish, inspect:

```text
outputs/ddp-training-1gcd/run_summary.json
outputs/ddp-training-8gcd-single-node/run_summary.json
outputs/ddp-training-*/raw/metrics_rank*.json
```

## What To Compare

Compare:

- total throughput
- mean rank throughput
- min/max rank throughput
- data wait fraction
- checkpoint time on rank 0
- placement files

## When This Pattern Is Appropriate

Use DDP-style scaling when ranks cooperate on the same training job and synchronize model updates.

Do not use DDP just to process independent records. For independent records, use the batch inference job-array example.

