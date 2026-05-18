# Scaling-Aware AI on LUMI: When Should You Use More GPUs?

A hands-on tutorial for diagnosing whether an AI workload should scale from 1 GCD to 8 GCDs to multiple LUMI-G nodes.

You have an AI workload. You can run it on 1 GCD, 8 GCDs on one full LUMI-G node, or 16+ GCDs across nodes.

Should you scale?

This repository helps AI and HPC users answer that question with evidence. The goal is not to benchmark LUMI. The goal is to spend small jobs to avoid wasting large jobs: run controlled experiments, collect evidence, diagnose the bottleneck, and produce a defensible scale decision.

Use this repo when you want to:

- test whether a workload benefits from moving from 1 GCD to a full LUMI-G node
- check whether multi-node execution adds value beyond one node
- diagnose data starvation, rank imbalance, launch/placement mistakes, and poor shard distribution
- compare distributed training with job-array style batch processing
- write a short decision record before requesting or repeating larger jobs

## Recommendation Categories

Every experiment should end with one of these recommendations:

| Recommendation | Meaning |
|---|---|
| `GO` | Scale to the next rung. Useful throughput improves with acceptable efficiency and GPU-hour cost. |
| `NO-GO` | Stay at the current rung. Extra GCDs do not buy enough useful throughput. |
| `FIX-FIRST` | Do not scale yet. A bottleneck is hiding the true scaling behavior. |
| `ARRAY` | Do not use distributed training. Use independent workers or Slurm job arrays. |
| `MEMORY-SCALE` | Scaling is justified for memory capacity, not throughput efficiency. |
| `INVALID-RUN` | Launch, rank count, node count, or placement is wrong. The result cannot be interpreted. |
| `MEASURE-MORE` | The evidence is too noisy or incomplete. |

Scaling is a decision, not a default.

Do not scale an unstable, data-starved, badly placed, or badly sharded workload. Scale only when the current rung provides evidence that the next rung is the right next experiment.

## LUMI-G And Billing Basics

In this README, **GCD** means the GPU-visible device that Slurm, HIP, and PyTorch treat as one GPU on LUMI-G.

A full LUMI-G node exposes 8 GPU-visible devices:

- 4 AMD MI250X modules
- 2 GCDs per MI250X
- 8 software-visible GCDs per node
- 56 CPU cores available to jobs
- Slingshot network connectivity for multi-node communication

PyTorch uses the CUDA-compatible API on LUMI even though the hardware is AMD/ROCm. It is normal for scripts to call `torch.cuda.device_count()`.

The tutorial ladder is:

```text
1 GCD             baseline
8 GCDs, 1 node    one full LUMI-G node
16 GCDs, 2 nodes  two LUMI-G nodes
```

The tutorial does not jump directly to 64 or 128 GCDs. If a workload cannot scale cleanly from 1 to 8 or from 8 to 16, scaling further usually wastes allocation or hides the real bottleneck.

Cost vocabulary:

- `GCD-hour`: one GCD allocated for one hour
- `LUMI GPU-hour`: one MI250X module allocated for one hour
- one MI250X has two GCDs
- therefore `1 LUMI GPU-hour = 2 GCD-hours`

Examples:

| Allocation | GCD-hours per wallclock hour | LUMI GPU-hours per wallclock hour |
|---|---:|---:|
| 1 GCD | 1 | 0.5 |
| 8 GCDs | 8 | 4 |
| 16 GCDs | 16 | 8 |

On `standard-g`, LUMI-G jobs allocate full nodes. One full LUMI-G node for one hour is billed as 4 LUMI GPU-hours. On `small-g` and `dev-g`, jobs can request GCD-level allocations, but CPU and memory requests can increase the charged amount.

## Setup

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

## Metrics Used In This Tutorial

Choose one useful work unit per workload:

- training: samples/sec, tokens/sec, step time
- batch inference: records/sec, completed outputs
- embeddings: documents/sec, chunks/sec

Use the same definitions throughout:

```text
throughput = useful_work / elapsed_seconds
speedup_1_to_8 = throughput_8gcd / throughput_1gcd
efficiency_1_to_8 = speedup_1_to_8 / 8
incremental_speedup_8_to_16 = throughput_16gcd / throughput_8gcd
incremental_efficiency_8_to_16 = incremental_speedup_8_to_16 / 2
logical_gcd_hours = gcd_count * walltime_hours
estimated_lumi_gpu_hours = logical_gcd_hours / 2
```

Also inspect:

- elapsed time
- rank elapsed spread
- shard elapsed imbalance
- data wait fraction
- communication fraction when available
- checkpoint or output-write cost when it affects end-to-end time

For synchronized training, the job is only as fast as the slowest rank. Summing local rank throughput can overstate performance if ranks finish at different times. Use the slowest rank elapsed time when computing job throughput:

```text
synchronized_throughput = total_samples_processed / max_rank_elapsed_time
```

## Part I: Prepare A Trustworthy Measurement

Before interpreting performance, prove that the launch is valid.

Run:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
```

After both jobs finish:

```bash
python scripts/validate_scaling_run.py
```

Expected valid result:

```text
VALIDATION_OK=1
baseline_world_size=1
single_node_world_size=8
baseline_gpu_visible_count=1
```

What to inspect:

```text
outputs/synthetic-*/raw/placement_rank*.json
```

A failed placement check blocks every scaling recommendation.

| Symptom | Recommendation | Reason |
|---|---|---|
| missing rank files | `INVALID-RUN` | not all ranks launched or wrote metrics |
| wrong world size | `INVALID-RUN` | the run shape is not what was requested |
| wrong node count | `INVALID-RUN` | single-node or multi-node interpretation is invalid |
| duplicate or suspicious device mapping | `INVALID-RUN` | rank-to-GCD placement may be wrong |

Only continue after validation passes.

## Part II: Scale To One Full LUMI-G Node

The first central question is:

> Does this workload benefit from moving from 1 GCD to all 8 GCDs on one LUMI-G node?

Run:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
```

After both jobs finish:

```bash
python scripts/compare_scaling.py
python scripts/build_lab_report.py
```

Read:

```text
outputs/scaling_report.md
outputs/lumi_hands_on_lab_report.md
outputs/synthetic-1gcd/run_summary.json
outputs/synthetic-8gcd-single-node/run_summary.json
```

Interpretation:

| Observation | Recommendation | Next Action |
|---|---|---|
| high 1-to-8 efficiency and valid placement | `GO` | try the 16-GCD multi-node rung |
| low 1-to-8 efficiency with valid placement | `FIX-FIRST` or `NO-GO` | diagnose data wait, work size, communication, or imbalance |
| noisy baseline | `MEASURE-MORE` | increase measured steps, exclude warmup, repeat |
| invalid placement | `INVALID-RUN` | fix launch before interpreting throughput |

Poor 1-to-8 efficiency can have several causes. The next sections deliberately create and fix common ones.

## Part III: Diagnose Poor Single-Node Scaling

### Challenge A: Data Starvation

What we break:

We make the input pipeline too slow, so ranks spend too much time waiting for data.

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

Evidence:

- high `mean_data_wait_fraction`
- GPU compute is not the dominant part of the step
- throughput improves when wait is reduced

Recommendation:

```text
FIX-FIRST
```

Real fixes:

- preprocess expensive transforms before training
- avoid many-small-file reads in the hot path
- shard data so every rank reads independent work
- tune dataloader workers and prefetching
- cache reusable transformed inputs

LUMI-specific CPU note:

On a full 8-GCD LUMI-G node, a practical upper bound is 56 allocatable CPU cores. For 8 ranks per node, 7 CPUs per rank is a sensible default. Requesting too many CPUs per rank can hurt scheduling or oversubscribe the node.

### Challenge B: Too Little Work Per GCD

What breaks:

The model, batch, sequence length, or synthetic workload is too small, so launch overhead and synchronization dominate useful compute.

How to pinpoint it:

- low compute time per step
- high synchronization overhead relative to compute
- very small per-rank batch
- efficiency improves when problem size increases

Fixes:

- increase per-rank batch if memory allows
- use gradient accumulation
- increase sequence length or problem size for the benchmark
- reduce unnecessary synchronization
- avoid tiny kernels and excessive Python-side step overhead

Recommendation:

```text
FIX-FIRST
```

### Challenge C: Binding And Topology Problems

What breaks:

Ranks run with poor CPU/GCD locality or inconsistent binding.

This is a useful distinction:

```text
A run can be valid but still poorly bound.
```

How to pinpoint it:

- placement validation passes
- no missing ranks
- no duplicate GCDs
- per-rank throughput varies strongly
- CPU affinity differs across ranks
- slow ranks repeat by local rank or socket

Fixes:

- use explicit CPU binding
- use local rank to select the local GCD
- use the LUMI `srun` rank-launch pattern
- avoid letting all ranks accidentally contend for the same device

Recommendation:

```text
FIX-FIRST
```

### Challenge D: Load Or Shard Imbalance

What breaks:

Different ranks or shards receive different amounts of work. In synchronized jobs, the slowest rank controls step time. In job arrays, the slowest shard controls walltime.

Run the imbalanced job-array case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_imbalanced.yaml \
  jobs/run_batch_inference_array_config.sh
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_imbalanced.yaml
```

Run the balanced case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_balanced.yaml \
  jobs/run_batch_inference_array_config.sh
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_balanced.yaml
python scripts/build_lab_report.py
```

The imbalanced and balanced inputs contain the same total `work_units`. Only the shard distribution changes.

Read:

```text
outputs/bottleneck-job-array-imbalanced/run_summary.json
outputs/solution-job-array-balanced/run_summary.json
outputs/*job-array*/raw/summary_shard*.json
```

Evidence:

- high `shard_elapsed_imbalance_ratio`
- high `max_shard_elapsed_seconds`
- low `throughput_records_per_sec_by_max_elapsed`
- uneven per-shard `work_units_total`

Fixes:

- shard by estimated token count, image size, or document length
- spread known-heavy records across shards
- use more smaller shards than workers
- bucket variable-length samples
- avoid rank-0-only work inside measured loops

Recommendation:

```text
FIX-FIRST
```

## Part IV: Scale Across Nodes

The second central question is:

> Does moving from one LUMI-G node to two nodes add enough useful throughput?

Run:

```bash
sbatch jobs/run_16gcd_two_node.sh
```

After the job finishes:

```bash
python scripts/compare_scaling.py
python scripts/validate_scaling_run.py
python scripts/build_lab_report.py
```

Read:

```text
outputs/scaling_report.md
outputs/synthetic-16gcd-two-node/run_summary.json
outputs/lumi_hands_on_lab_report.md
```

Key signal:

```text
incremental_speedup_8_to_16 = throughput_16gcd / throughput_8gcd
incremental_efficiency_8_to_16 = incremental_speedup_8_to_16 / 2
```

Interpretation:

| Observation | Recommendation | Reason |
|---|---|---|
| high 8-to-16 incremental efficiency | `GO` | the second node adds useful throughput |
| low 8-to-16 incremental efficiency | `NO-GO` | the second node adds little value |
| valid 1-to-8 but poor 8-to-16 | `NO-GO` or `FIX-FIRST` | inter-node communication or synchronization may dominate |
| invalid node or rank count | `INVALID-RUN` | the multi-node result cannot be interpreted |

A workload can scale well from 1 to 8 and still fail from 8 to 16. The 8-to-16 incremental efficiency is the key multi-node signal.

Possible fixes for poor multi-node scaling:

- increase compute per synchronization
- use gradient accumulation
- increase per-rank batch if possible
- reduce communication volume
- overlap communication with computation
- avoid unnecessary barriers

## Part V: Recognize Non-Throughput Scaling Cases

### Checkpoint And Filesystem Overhead

A run can have good steady-state throughput but poor end-to-end throughput because checkpointing or output writes dominate.

How to pinpoint it:

- steady-state step time looks good
- end-to-end throughput is much worse
- all ranks write files
- checkpoint time grows with rank count
- many small files are created

Fixes:

- checkpoint less frequently
- use intentional sharded checkpointing
- avoid all ranks writing duplicate full checkpoints
- reduce metadata pressure
- measure steady-state and end-to-end throughput separately

Recommendation:

```text
FIX-FIRST
```

### Job Arrays Instead Of Distributed Training

Use job arrays when records are independent:

- inference over unrelated records
- preprocessing
- evaluation cases
- hyperparameter sweeps
- many unrelated small experiments

Do not use DDP when no rank needs another rank's result.

Recommendation:

```text
ARRAY
```

### Memory-Driven Scaling

Sometimes scaling is justified because the workload does not fit, not because throughput efficiency is high.

Examples:

- model weights do not fit on one GCD
- optimizer state is too large
- activation memory is too high
- sequence length or microbatch requires sharding

Try before scaling blindly:

- mixed precision
- activation checkpointing
- smaller microbatch
- gradient accumulation
- optimizer sharding
- FSDP or ZeRO-style sharding

Recommendation:

```text
MEMORY-SCALE
```

`MEMORY-SCALE` is not the same as `GO`. It means scaling may be necessary for capacity, but you should not claim strong throughput scaling unless the measurements support it.

## Final Recommendation Report

Generate or refresh the report:

```bash
python scripts/build_lab_report.py
```

Then fill in:

```text
templates/scale-decision-record.md
```

A good report includes:

```text
Workload:
Useful work unit:
Validation result:
1-GCD throughput:
8-GCD throughput:
16-GCD throughput:
1-to-8 speedup and efficiency:
8-to-16 incremental speedup and efficiency:
GCD-hours and estimated LUMI GPU-hours:
Data wait:
Rank or shard imbalance:
Recommendation:
Reason:
Next action:
```

Good recommendation example:

```text
Recommendation: GO
Reason:
The workload has valid placement, stable measurements, good single-node efficiency,
and good incremental multi-node efficiency.
Next action:
Run the production workload at 16 GCDs, then repeat this decision process before scaling further.
```

Bad recommendation example:

```text
Recommendation: FIX-FIRST
Reason:
8-GCD efficiency is poor and data wait is high.
Placement validation passed, so this is not a launch failure.
The input pipeline is the dominant bottleneck.
Next action:
Fix dataloading and repeat the 8-GCD run.
Do not run the 16-GCD test yet.
```

## Decision Tree

```text
Did validation pass?
  no -> INVALID-RUN
Is the baseline stable?
  no -> MEASURE-MORE
Is the useful work unit declared?
  no -> MEASURE-MORE
Is the workload independent?
  yes -> ARRAY
Does the model require more memory?
  yes -> MEMORY-SCALE
Is data wait high?
  yes -> FIX-FIRST
Is rank or shard imbalance high?
  yes -> FIX-FIRST
Is 1-to-8 efficiency poor?
  yes -> FIX-FIRST or NO-GO
Is 8-to-16 incremental efficiency poor?
  yes -> NO-GO unless memory requires scaling
Otherwise:
  GO
```

## Repository Layout

```text
configs/    YAML configs for each lab
examples/   JSONL inputs for bottleneck labs
jobs/       Slurm job scripts
logs/       Slurm output directory
scripts/    Workloads, collectors, validators, report builder
templates/  Decision record template
```
