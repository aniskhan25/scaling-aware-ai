# Scaling-Aware AI on LUMI

A small hands-on lab for deciding when to scale an AI workload on LUMI-G.

The repo runs three kinds of experiments:

- synthetic scaling from 1 GCD to 8 GCDs to 16 GCDs
- DDP training with induced data wait and a reduced-wait fix
- batch inference with imbalanced and balanced job-array shards

The point is not to benchmark LUMI. The point is to collect enough evidence to decide whether to scale up, fix the current scale, or use job arrays instead of distributed collectives.

## Setup

The Slurm jobs are configured for:

```text
project_462000131
/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

Run commands from the repository root on LUMI:

```bash
cd /path/to/scaling-aware-ai
```

Slurm logs are written to `logs/`. Run artifacts are written to `outputs/`.

For the detailed walkthrough and interpretation notes, start with:

```text
guide/00-hands-on-lumi-lab.md
```

## Run The Scaling Ladder

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

Use the larger scale only when placement is valid and efficiency remains high enough to justify the extra GPU-hours.

## Run The DDP Data-Wait Lab

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_bottleneck.yaml \
  jobs/run_ddp_8gcd_config.sh

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

If data wait is high, fix the input pipeline before scaling further.

## Run The Job-Array Imbalance Lab

The imbalanced and balanced inputs contain the same total `work_units`. Only the shard distribution changes.

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_imbalanced.yaml \
  jobs/run_batch_inference_array_config.sh

python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_imbalanced.yaml

sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_balanced.yaml \
  jobs/run_batch_inference_array_config.sh

python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_balanced.yaml

python scripts/build_lab_report.py
```

Read:

```text
outputs/bottleneck-job-array-imbalanced/run_summary.json
outputs/solution-job-array-balanced/run_summary.json
outputs/lumi_hands_on_lab_report.md
```

For independent records, balance shards and use job arrays before adding distributed machinery.

## Decision Record

Fill in:

```text
templates/scale-decision-record.md
```

Minimum fields:

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

## Repository Layout

```text
configs/    YAML configs for each lab
examples/   JSONL inputs for bottleneck labs
guide/      Detailed lab guide and scaling background
jobs/       Slurm job scripts
logs/       Slurm output directory
scripts/    Workloads, collectors, validators, report builder
templates/  Decision record template
```

## Core Rule

Do not scale an unstable, data-starved, or badly sharded workload. Scale only after the current rung produces evidence that the next rung is the right next experiment.
