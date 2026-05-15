# Scaling-Aware AI on LUMI

A hands-on LUMI-G tutorial for making evidence-based scaling decisions for AI workloads.

This repository helps AI and HPC users answer a practical question:

> Should this workload use more LUMI-G GPUs, or should I fix the workload first?

The goal is not to benchmark LUMI. The goal is to spend small jobs to avoid wasting large jobs: run controlled experiments, collect evidence, diagnose the bottleneck, and make a defensible scale decision.

Use this repo when you want to:

- test whether a workload benefits from moving from 1 GCD to a full LUMI-G node
- check whether multi-node execution adds value beyond a single node
- diagnose data starvation, rank imbalance, launch/placement mistakes, and poor shard distribution
- compare distributed training with job-array style batch processing
- produce a short decision record before requesting or repeating larger jobs

## Core Rule

Scaling is a decision, not a default.

Do not scale an unstable, data-starved, badly placed, or badly sharded workload. Scale only when the current rung provides evidence that the next rung is the right next experiment.

A larger job is justified when it improves useful throughput with acceptable efficiency and GPU-hour cost. A larger job is not justified when the bottleneck is input data, launch configuration, rank placement, shard imbalance, or insufficient work per GPU.

## Objective

By the end of this tutorial, you should be able to produce a defensible scale decision for an AI workload on LUMI-G.

A good decision answers:

- What is the useful work unit: samples, tokens, records, documents, or chunks?
- What is the baseline throughput on the smallest useful run?
- Does a full LUMI-G node improve throughput enough to justify the extra GPU-hours?
- Does a multi-node run add value beyond single-node scaling?
- Is the workload bottlenecked by compute, data wait, communication, placement, or shard imbalance?
- Should the workload use distributed training, independent workers, or Slurm job arrays?
- What evidence would justify moving to the next scale?

The expected output is not just a successful Slurm job. The expected output is a scaling report and a decision record.

## LUMI Setup

Run from the repository root on LUMI:

```bash
cd /path/to/scaling-aware-ai
```

The Slurm jobs are configured for:

```text
project_462000131
/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

Slurm logs are written to `logs/`. Run artifacts are written to `outputs/`.

Full-node and multi-node jobs use the LUMI `srun` pattern:

```text
--ntasks-per-node=8
--cpus-per-task=7
--mem-per-gpu=60G
srun --cpu-bind=v,mask_cpu=<LUMI masks>
```

Each task exports:

```bash
RANK=$SLURM_PROCID
LOCAL_RANK=$SLURM_LOCALID
```

## LUMI-G Mental Model

A full LUMI-G node exposes 8 GPU-visible devices to PyTorch:

- 4 AMD MI250X modules
- 2 GCDs per MI250X
- 8 software-visible GCDs per node
- 56 CPU cores available to jobs
- Slingshot network connectivity for multi-node communication

On LUMI, PyTorch uses the CUDA-compatible API even though the hardware is AMD/ROCm. It is normal for scripts to call `torch.cuda.device_count()`.

In this README, GCD means the GPU-visible device that Slurm, HIP, and PyTorch treat as one GPU on LUMI-G.

The basic diagnostic ladder is:

```text
1 GCD             baseline compute and local overhead
8 GCDs, 1 node    intra-node scaling
16 GCDs, 2 nodes  intra-node plus inter-node scaling
```

If 8-GCD scaling is weak, fix the local workload, launch, placement, or data path before interpreting multi-node behavior. If 8-GCD scaling is good but 16-GCD scaling drops sharply, inter-node communication or synchronization is more likely.

## Metrics

Use one useful work unit per workload:

- training: samples/sec, tokens/sec, step time
- batch inference: records/sec, completed outputs
- embeddings: documents/sec, chunks/sec

Report these together:

```text
speedup = throughput_target / throughput_baseline
efficiency = speedup / (target_world_size / baseline_world_size)
gpu_hours = number_of_gcds * walltime_hours
```

Raw throughput can improve while the run becomes wasteful. Efficiency and GPU-hours make the cost visible.

Also inspect per-rank or per-shard variance. A distributed job often waits for the slowest rank or shard.

## Scaling Ladder

| Stage | Question | Evidence | Move Up When | Stop Or Fix When |
|---|---|---|---|---|
| 0. Define workload | What is useful work? | samples/sec, records/sec, tokens/sec | the metric matches the goal | the metric is only "job completed" |
| 1. Single GCD | Is the smallest useful run healthy? | throughput, memory, data wait, placement | throughput is stable | input, memory, correctness, or startup is unstable |
| 2. Full node | Does 8 GCD help? | 1 vs 8 GCD speedup, efficiency, rank placement | efficiency is acceptable | placement is wrong or efficiency collapses |
| 3. Multi-node | Does the networked run still help? | 8 vs 16 GCD speedup, efficiency, host/rank spread | single-node is strong and multi-node adds value | inter-node communication dominates |
| 4. Pattern choice | Is distributed execution needed? | dependency structure | ranks must synchronize | records are independent |
| 5. Decision | Is this scale worth repeating? | report, logs, outputs, GPU-hours | the decision is documented | cost or risk exceeds benefit |

## Lab 1: Synthetic Scaling Ladder

Question:

> Does this workload justify moving from 1 GCD, to one full LUMI-G node, to two nodes?

Submit:

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

Interpretation:

| Observation | Meaning | Next Action |
|---|---|---|
| 8-GCD and 16-GCD both scale well | workload may justify larger staged runs | increase scale gradually |
| 8-GCD good, 16-GCD poor | network or cross-node communication is limiting | inspect communication pattern and workload size |
| 16-GCD barely improves over 8-GCD | multi-node is not justified for this workload size | stay single-node or increase per-rank work |
| rank or node count is wrong | result is invalid | fix launch before interpreting performance |
| per-rank spread is large | imbalance, placement, or CPU affinity may matter | inspect placement files |

## Lab 2: DDP Data Starvation

Question:

> What happens when synchronized training waits for data instead of doing GPU work?

Run the induced bottleneck:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_bottleneck.yaml \
  jobs/run_ddp_8gcd_config.sh
```

Run the reduced-wait case:

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

Focus on:

- `total_throughput_samples_per_sec`
- `mean_data_wait_fraction`
- `max_data_wait_fraction`
- `rank_elapsed_spread_seconds`

Real fixes this represents:

- preprocess expensive transforms before training
- avoid many-small-file reads in the hot path
- shard data so every rank reads independent work
- tune dataloader workers and prefetching
- cache reusable transformed inputs

Decision rule:

If data wait is high, fix the input pipeline before requesting a larger LUMI-G job.

## Lab 3: Job-Array Shard Imbalance

Question:

> For independent batch work, does a distributed launch help, or is the slowest shard the real limiter?

The imbalanced and balanced inputs contain the same total `work_units`. Only the shard distribution changes.

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

Focus on:

- `max_shard_elapsed_seconds`
- `shard_elapsed_imbalance_ratio`
- `throughput_records_per_sec_by_max_elapsed`
- per-shard `work_units_total`

Real fixes this represents:

- shard by estimated token count, image size, or document length
- spread known-heavy records across shards
- use more smaller shards than workers
- retry failed shards independently
- write one output and one summary per shard

Decision rule:

For independent records, fix sharding and use job arrays before reaching for distributed collectives.

## Workload Pattern Choice

Use distributed training when:

- ranks cooperate on one model update stream
- gradients or parameters synchronize
- every step depends on all ranks
- global batch behavior matters

Use job arrays or independent workers when:

- records are independent
- outputs can be merged later
- failed shards can be retried
- no rank needs another rank's result during processing

| Workload | Better First Pattern | Why |
|---|---|---|
| DDP training | full-node DDP | gradients synchronize each step |
| corpus embedding | job array or independent workers | documents are independent |
| batch evaluation | job array | cases can be scored separately |
| large model that does not fit | sharding or model parallelism | memory, not simple throughput, is the blocker |
| online serving | replicas and batching | latency and queueing matter |

## Report And Decision Record

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

A useful decision says either:

- scale up: the next rung improves useful throughput with acceptable efficiency and GPU-hour cost
- stay smaller: the current scale is efficient, and larger runs add little value
- fix first: data wait, placement, launch overhead, communication, or imbalance hides the real scaling behavior
- use job arrays: records are independent, and distributed collectives add unnecessary synchronization

## Repository Layout

```text
configs/    YAML configs for each lab
examples/   JSONL inputs for bottleneck labs
jobs/       Slurm job scripts
logs/       Slurm output directory
scripts/    Workloads, collectors, validators, report builder
templates/  Decision record template
```
