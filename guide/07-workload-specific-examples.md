# 7. Workload-Specific Examples

The examples in this guide demonstrate two different scaling problems:

1. synchronized training, where ranks must communicate
2. independent batch processing, where ranks do not need to communicate

This distinction matters more than the launch command. Using distributed collectives for independent work adds failure modes without adding value.

## Example A: Synchronized DDP Training

Use this pattern when the model update depends on all ranks.

The DDP example asks:

> Does synchronized data-parallel training improve useful training throughput on one full LUMI-G node?

It deliberately records more than just samples/sec:

- rank placement
- world size
- local and global batch size
- per-rank throughput
- loss
- data wait fraction
- checkpoint time on rank 0

These metrics answer different questions.

| Observation | What It Tells You |
|---|---|
| local throughput is good on 1 GCD | baseline compute path is viable |
| 8-GCD throughput improves with acceptable efficiency | DDP is probably useful |
| high data wait fraction | input delivery would limit real training |
| high checkpoint time | end-to-end walltime may not improve as much as step time |
| large per-rank throughput spread | placement, CPU affinity, or imbalance needs inspection |

### When To Move Up

Move from 1 GCD to 8 GCDs when:

- the single-GCD run is stable
- the model fits comfortably
- the per-step workload is large enough
- faster training turnaround matters

Do not move to multi-node yet if:

- the 8-GCD result is weak
- placement metadata is missing or inconsistent
- throughput is dominated by simulated or real data wait
- checkpointing dominates wall-clock time

### Runbook

```bash
sbatch jobs/run_ddp_1gcd.sh
sbatch jobs/run_ddp_8gcd_single_node.sh
```

Read:

```text
outputs/ddp-training-1gcd/run_summary.json
outputs/ddp-training-8gcd-single-node/run_summary.json
outputs/ddp-training-*/raw/metrics_rank*.json
outputs/ddp-training-*/raw/placement_rank*.json
```

### Practical Challenge To Demonstrate

Set `synthetic_data_wait_seconds` in `configs/ddp-training/*.yaml` to a small nonzero value and rerun.

Expected lesson:

- GPU scaling may look worse even though the launch is correct.
- The bottleneck is now input wait, not DDP itself.
- Scaling up compute does not fix a starved training loop.

## Example B: Independent Batch Inference

Use this pattern when records can be processed independently.

The batch inference example asks:

> Can independent work be split into restartable shards without distributed communication?

It records:

- records written
- shard index
- shard elapsed time
- max shard elapsed time
- records/sec
- completed shard summaries

These are the right metrics for independent throughput. DDP-style world-size and all-reduce metrics are the wrong abstraction.

| Observation | What It Tells You |
|---|---|
| all shards complete | the work is shardable and restartable |
| one shard is much slower | sharding is imbalanced |
| records are missing | aggregation or retry logic is not trustworthy |
| records/sec improves with more array tasks | independent parallelism is effective |
| startup/model-load time dominates | batching or larger shards may matter more than more tasks |

### When To Use This Instead Of DDP

Use a job array when:

- each input record can be processed alone
- no rank needs another rank's result
- partial failure should be retryable
- outputs can be merged after the run

Do not use a job array when:

- ranks must synchronize gradients
- one model state must be updated continuously
- the model must be sharded across devices to fit
- online latency and request routing dominate the problem

### Runbook

```bash
sbatch jobs/run_batch_inference_array.sh
python scripts/collect_batch_inference.py --config configs/batch-inference/job_array.yaml
```

Read:

```text
outputs/batch-inference-array/run_summary.json
outputs/batch-inference-array/raw/summary_shard*.json
outputs/batch-inference-array/raw/outputs_shard*.jsonl
```

### Practical Challenge To Demonstrate

Change the job-array size or the input file size and compare max shard elapsed time.

Expected lesson:

- Total records/sec is limited by the slowest shard.
- More array tasks help only if the work is balanced and startup overhead is not dominant.
- Independent parallelism needs a merge and validation step just as much as distributed training needs placement validation.

## Decision Summary

| If You Observe | Prefer |
|---|---|
| gradients synchronize every step | DDP or another distributed training strategy |
| records are independent | job array or independent workers |
| model does not fit on one device | sharding or model parallelism |
| real data is slow but synthetic data is fast | data pipeline work |
| single-node is weak | do not move to multi-node yet |

## Practical Rule

Match the scaling pattern to the dependency structure of the workload.

The wrong abstraction can make a workload larger, slower, and harder to debug at the same time.
