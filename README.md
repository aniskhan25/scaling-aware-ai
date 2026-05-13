# Scaling-Aware AI on LUMI

This is a practical guide for deciding **whether a workload should scale up on LUMI-G**, not just how to launch a larger job.

The central idea is simple:

> Move up the scale ladder only when the current level has produced evidence that scaling is the right next intervention.

Many workloads should not go directly from one visible device to multi-node execution. Some need a better single-GCD baseline. Some need data pipeline work. Some need a full-node run. Some should use job arrays instead of distributed collectives. This guide is about making that distinction from observations.

## The Scaling Ladder

Use the ladder as a decision process:

| Stage | Question | Evidence To Collect | Move Up When | Stop Or Fix When |
|---|---|---|---|---|
| 0. Define the workload | What is the useful work unit? | samples/sec, tokens/sec, records/sec, latency, GPU-hours | the metric matches the objective | the metric is vague or only reports "job completed" |
| 1. Single-GCD baseline | Is the smallest useful run healthy? | steady-state throughput, memory, data wait, logs | GPU work is stable and repeatable | data loading, memory, startup, or correctness is unstable |
| 2. Full-node test | Does one LUMI-G node improve useful throughput? | 1 vs 8 GCD speedup, efficiency, rank placement | throughput improves enough for the cost | rank placement is wrong or efficiency collapses |
| 3. Multi-node test | Does networked scaling add value? | 8 vs 16 GCD speedup, efficiency, per-rank variance | single-node is already strong and multi-node adds useful throughput | inter-node communication dominates |
| 4. Workload pattern choice | Is distributed execution the right abstraction? | dependency structure of the work | ranks must synchronize | records are independent and a job array is simpler |
| 5. Production plan | Is the chosen scale worth repeating? | GPU-hours, walltime, artifacts, restart plan | the decision is documented and reproducible | cost or operational risk exceeds the benefit |

The guide includes runnable examples, but the examples are there to support this reasoning. They are not the point by themselves.

The code is intentionally minimal. Each script should make one concept visible: placement, throughput, data wait, communication, or shard imbalance. Avoid adding framework-like abstractions unless they make the bottleneck easier to see.

## Start With These Chapters

1. [Introduction](guide/01-introduction.md)
2. [LUMI-G mental model](guide/02-lumi-g-mental-model.md)
3. [Scaling metrics](guide/03-scaling-metrics.md)
4. [Scaling decision ladder](guide/04-synthetic-scaling-ladder.md)
5. [Workload taxonomy](guide/05-workload-taxonomy.md)
6. [Data pipeline scaling](guide/06-data-pipeline-scaling.md)
7. [Workload-specific examples](guide/07-workload-specific-examples.md)
8. [Bottleneck demonstrations](guide/08-bottleneck-demonstrations.md)

## What The Examples Demonstrate

The examples are intentionally small. Their purpose is to expose scaling decisions:

- **Synthetic ladder**: separates launch correctness, placement, compute, and collective communication.
- **DDP training**: shows why synchronized training needs rank-level metrics, data wait checks, and checkpoint timing.
- **Batch inference job array**: shows when independent records should avoid distributed collectives entirely.
- **Bottleneck labs**: induce data starvation, shard imbalance, and too-small-workload behavior, then show what evidence confirms the fix.

## Minimal Runbook

Use this after reading the decision ladder.

Synthetic ladder:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
sbatch jobs/run_16gcd_two_node.sh
python scripts/compare_scaling.py
python scripts/validate_scaling_run.py
```

DDP training:

```bash
sbatch jobs/run_ddp_1gcd.sh
sbatch jobs/run_ddp_8gcd_single_node.sh
```

Batch inference job array:

```bash
sbatch jobs/run_batch_inference_array.sh
python scripts/collect_batch_inference.py --config configs/batch-inference/job_array.yaml
```

Bottleneck demonstrations:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_bottleneck.yaml jobs/run_ddp_8gcd_config.sh
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_reduced.yaml jobs/run_ddp_8gcd_config.sh
```

Before running, replace `#SBATCH --account=project_XXXXXXXXX` in the job scripts.

## Outputs To Read

Do not stop at "the job ran." Read the artifacts:

```text
outputs/scaling_report.md
outputs/scaling_report.json
outputs/*/run_summary.json
outputs/*/environment.json
outputs/*/raw/placement_rank*.json
outputs/*/raw/metrics_rank*.json
```

The important questions are:

- Did the observed rank and node counts match the intended launch?
- Did per-rank throughput look balanced?
- Did speedup justify the larger world size?
- Did efficiency collapse at single-node or only at multi-node?
- Is the workload synchronized training, independent batch work, or something else?

Use [scale-decision-record.md](templates/scale-decision-record.md) to turn those observations into a documented stop/go decision.

## Core Rule

Do not scale an unstable or undersized workload.

A larger run is useful only if it improves useful throughput enough to justify the communication, launch complexity, queue time, and GPU-hour cost.

## Repository Layout

```text
scaling-aware-ai/
  README.md
  ROADMAP.md
  CONTRIBUTING.md
  configs/
  docs/
  examples/
  guide/
  jobs/
  scripts/
  templates/
```

## Project Notes

- [Product brief](docs/product-brief.md)
- [Content architecture](docs/content-architecture.md)
- [Technical scope](docs/technical-scope.md)
- [Roadmap](ROADMAP.md)
- [Contributing guide](CONTRIBUTING.md)
