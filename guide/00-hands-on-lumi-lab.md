# 0. Hands-On LUMI Scaling Lab

This is the primary path through the repository.

The goal is to run small jobs on LUMI-G, observe scaling behavior, induce common failures, apply the matching fix, and produce a decision record. The conceptual chapters are supporting material for this lab.

## Prerequisites

Work on LUMI from the repository root:

```bash
cd /path/to/scaling-aware-ai
```

The job files are configured for the Slurm account `project_462000131` and the LUMI MultiTorch container:

```bash
export CONTAINER=/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

The job scripts load the LUMI AI bindings module and run the Python scripts inside `CONTAINER`.

For full-node and multi-node runs, the scripts use one Slurm task per GPU-visible GCD:

```text
--ntasks-per-node=8
--cpus-per-task=7
--mem-per-gpu=60G
srun --cpu-bind=v,mask_cpu=<LUMI masks>
```

Each task exports `RANK=$SLURM_PROCID` and `LOCAL_RANK=$SLURM_LOCALID` before starting Python.

## Lab 1: Scaling Ladder

Question:

> Does this workload justify moving from 1 GCD, to one full LUMI-G node, to two nodes?

Submit the three runs:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
sbatch jobs/run_16gcd_two_node.sh
```

After all three jobs finish:

```bash
python scripts/compare_scaling.py
python scripts/validate_scaling_run.py
python scripts/build_lab_report.py
```

Read:

```text
outputs/scaling_report.md
outputs/lumi_hands_on_lab_report.md
outputs/synthetic-*/run_summary.json
outputs/synthetic-*/raw/placement_rank*.json
```

What to observe:

| Evidence | Meaning |
|---|---|
| `world_size_matches_expected=true` | the launch shape is valid |
| `node_count_matches_expected=true` | the run used the intended node count |
| 8-GCD efficiency is strong | single-node scale-up is plausible |
| 16-GCD efficiency adds little over 8 GCDs | multi-node is not justified for this workload size |
| rank elapsed spread is large | inspect placement, CPU binding, or per-rank imbalance |

Decision:

Move up only when the larger run gives enough useful throughput for the additional GPU-hours and operational complexity.

## Lab 2: Induce And Fix DDP Data Starvation

Question:

> What happens when synchronized training waits for data instead of doing GPU work?

Run the induced bottleneck:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_bottleneck.yaml \
  jobs/run_ddp_8gcd_config.sh
```

Run the resolution case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_reduced.yaml \
  jobs/run_ddp_8gcd_config.sh
```

After both jobs finish:

```bash
python scripts/build_lab_report.py
```

Read:

```text
outputs/bottleneck-ddp-data-wait/run_summary.json
outputs/solution-ddp-data-wait-reduced/run_summary.json
outputs/lumi_hands_on_lab_report.md
```

What to observe:

| Evidence | Meaning |
|---|---|
| high `mean_data_wait_fraction` | the ranks are waiting on input |
| lower wait fraction in the resolution case | data delivery is the bottleneck being fixed |
| improved throughput after reducing wait | scale-up was not the first fix |

Real fixes this represents:

- preprocess expensive transforms before training
- avoid many-small-file reads in the hot path
- shard data so every rank reads independent work
- tune dataloader workers and prefetching
- cache reusable transformed inputs

Decision:

If data wait is high, fix the input pipeline before requesting a larger LUMI-G job.

## Lab 3: Induce And Fix Job-Array Imbalance

Question:

> For independent batch work, does a distributed launch help, or is the slowest shard the real limiter?

The imbalanced and balanced inputs contain the same total `work_units`. The difference is only how that work is distributed across eight array shards.

Run the imbalanced case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_imbalanced.yaml \
  jobs/run_batch_inference_array_config.sh
```

After the array finishes:

```bash
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_imbalanced.yaml
```

Run the balanced case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_balanced.yaml \
  jobs/run_batch_inference_array_config.sh
```

After the array finishes:

```bash
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_balanced.yaml
python scripts/build_lab_report.py
```

Read:

```text
outputs/bottleneck-job-array-imbalanced/run_summary.json
outputs/solution-job-array-balanced/run_summary.json
outputs/*job-array*/raw/summary_shard*.json
outputs/lumi_hands_on_lab_report.md
```

What to observe:

| Evidence | Meaning |
|---|---|
| high `shard_elapsed_imbalance_ratio` | one shard dominates walltime |
| lower max shard elapsed after balancing | the fix reduced the tail shard |
| better throughput by slowest shard | total walltime improved without collectives |

Real fixes this represents:

- shard by estimated tokens, image size, or document length
- use more smaller shards than workers
- spread known-heavy records across shards
- write one output and summary per shard
- retry failed shards independently

Decision:

For independent records, fix sharding and use job arrays before reaching for distributed collectives.

## Lab 4: Write The Scale Decision

Generate or refresh the report:

```bash
python scripts/build_lab_report.py
```

Then fill in:

```text
templates/scale-decision-record.md
```

Minimum decision fields:

```text
Workload objective:
Useful metric:
Baseline result:
Single-node result:
Multi-node result:
Observed bottleneck:
Fix attempted:
Result after fix:
Chosen scale:
What would justify moving up:
```

The lab is complete when the decision says either:

- scale up, with evidence from the current rung
- stay smaller, with the bottleneck and next fix identified
