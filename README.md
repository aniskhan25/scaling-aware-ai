# Scaling-Aware AI on LUMI: When Should You Use More GPUs?

A hands-on tutorial for deciding whether an AI workload should scale from 1 GCD to a full LUMI-G node to multiple nodes.

> Should you scale?

This repository helps AI and HPC users answer that question with evidence. The goal is not to benchmark LUMI. The goal is to spend small jobs to avoid wasting large jobs: run controlled experiments, collect evidence, diagnose the bottleneck, and produce a defensible scale decision.

Use this repo when you want to:

- test whether a workload benefits from moving from 1 GCD to a full LUMI-G node
- check whether multi-node execution adds value beyond one node
- diagnose data starvation, rank imbalance, and poor shard distribution
- compare distributed training with job-array style batch processing
- print enough performance evidence to decide the next run

> [!IMPORTANT]
> Scaling is a decision, not a default. Do not scale an unstable, data-starved, badly placed, or badly sharded workload. Scale only when the current rung provides evidence that the next rung is the right next experiment.

## LUMI-G And Billing Basics

**GCD** means the GPU-visible device that Slurm, HIP, and PyTorch treat as one GPU on LUMI-G.

A full LUMI-G node exposes 8 GPU-visible devices:

- 4 AMD MI250X modules
- 2 GCDs per MI250X
- 8 software-visible GCDs per node
- 56 CPU cores available to jobs
- Slingshot network connectivity for multi-node communication

The tutorial ladder is:

```text
1 GCD             baseline
8 GCDs, 1 node    one full LUMI-G node
16 GCDs, 2 nodes  two LUMI-G nodes
```

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

The 1-GCD and job-array examples use `small-g`. Full-node and two-node examples use `standard-g`.

Slurm logs are written to `logs/`. Run artifacts are written to `outputs/`.

## Part I: Run The Baseline Jobs

> [!TIP]
> Start from a fresh `outputs/` directory when rerunning the full tutorial.

Submit the first two jobs:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
```

At the end of each Slurm log, the job prints a compact summary.

Expected 1-GCD output shape:

```text
RUN_SUMMARY=.../outputs/synthetic-1gcd/run_summary.json
WORLD_SIZE=1
NODES=1
RANKS=1
TOTAL_THROUGHPUT=...
RANK_ELAPSED_SPREAD=0.0000
```

Expected 8-GCD output shape:

```text
RUN_SUMMARY=.../outputs/synthetic-8gcd-single-node/run_summary.json
WORLD_SIZE=8
NODES=1
RANKS=8
TOTAL_THROUGHPUT=...
RANK_ELAPSED_SPREAD=...
```

## Part II: Scale To One Full LUMI-G Node

The first central question is:

> Does this workload benefit from moving from 1 GCD to all 8 GCDs on one LUMI-G node?

The default synthetic ladder is intentionally healthy. It is the reference case that proves the scaling calculations are working. If this run scales well, that does not mean every real workload should scale. It means this controlled workload has enough useful work per GCD and no obvious single-node bottleneck.

The later challenge runs deliberately create cases where the right action is to fix the workload, stay smaller, or use job arrays.

Use the completed jobs from Part I.

Print the scaling comparison:

```bash
python scripts/compare_scaling.py
```

The printed values are also saved in:

```text
outputs/synthetic-1gcd/run_summary.json
outputs/synthetic-8gcd-single-node/run_summary.json
```

The comparison uses:

```text
speedup_1_to_8 = throughput_8gcd / throughput_1gcd
efficiency_1_to_8 = speedup_1_to_8 / 8
```

> [!NOTE]
> These examples keep per-rank work fixed, so the global work per step grows with the number of GCDs. Treat the result as throughput scaling, not proof that a real training workload reaches the same quality faster.

Interpretation:

| Observation | What to do | Reason |
|---|---|---|
| high 1-to-8 efficiency | try the 16-GCD multi-node rung | the full node adds useful throughput |
| low 1-to-8 efficiency | diagnose the bottleneck before scaling | data wait, work size, communication, or imbalance may dominate |
| noisy baseline | repeat with more stable measurement | increase measured steps, exclude warmup, repeat |

Good scaling here is a checkpoint, not the end of the tutorial. It says the clean synthetic workload can move to the 16-GCD rung. The next sections deliberately create common bottlenecks so you can recognize when a real workload should not scale yet.

## Part III: Diagnose Bottlenecks Before Scaling

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

The demonstrated fix is the config change from:

```text
synthetic_data_wait_seconds: 0.050
```

to:

```text
synthetic_data_wait_seconds: 0.005
```

Everything else stays comparable, so the output shows what happens when the input wait is reduced.

Each job prints its summary path, throughput, rank elapsed spread, and data wait fraction at the end of the Slurm log.

Expected output shape:

```text
RUN_SUMMARY=.../outputs/bottleneck-ddp-data-wait/run_summary.json
WORLD_SIZE=8
NODES=1
RANKS=8
TOTAL_THROUGHPUT=...
RANK_ELAPSED_SPREAD=...
MEAN_DATA_WAIT_FRACTION=...
MAX_DATA_WAIT_FRACTION=...
```

The printed values are also saved in:

```text
outputs/bottleneck-ddp-data-wait/run_summary.json
outputs/solution-ddp-data-wait-reduced/run_summary.json
```

Evidence:

- high `mean_data_wait_fraction`
- GPU compute is not the dominant part of the step
- throughput improves when wait is reduced

`mean_data_wait_fraction` is the fraction of measured step time spent waiting for input data rather than doing useful compute.

> [!WARNING]
> The workload is input-pipeline limited. Do not add more GCDs until the data path can feed the current scale.

In a real workload, the same symptom can come from expensive CPU transforms, many small files, slow metadata access, too few dataloader workers, or missing prefetching/caching. The next action is to make the input path faster, then rerun the same 8-GCD comparison.

CPU allocation can also affect data wait. On a full 8-GCD LUMI-G node, 56 CPU cores are available to jobs, so 7 CPUs per rank is a sensible starting point for 8 ranks. Avoid requesting or spawning more CPU work than the node can support.

If data wait is low but 1-to-8 efficiency is still poor, the workload may simply be too small for 8 GCDs. Increase per-rank batch size, sequence length, or useful compute per step, then rerun the same comparison.

### Challenge B: Load Or Shard Imbalance

What breaks:

Different ranks or shards receive different amounts of work. In synchronized jobs, the slowest rank controls step time. In job arrays, the slowest shard controls walltime.

This example demonstrates the job-array case: the total work is the same, but one input distributes the work unevenly across shards and the other distributes it more evenly.

Run the imbalanced job-array case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_imbalanced.yaml \
  jobs/run_batch_inference_array_config.sh
```

After the array finishes, collect the shard summaries:

```bash
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_imbalanced.yaml
```

Run the balanced case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_balanced.yaml \
  jobs/run_batch_inference_array_config.sh
```

After the array finishes, collect the shard summaries:

```bash
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_balanced.yaml
```

The demonstrated fix is the input change from:

```text
examples/bottlenecks/data/imbalanced_requests.jsonl
```

to:

```text
examples/bottlenecks/data/balanced_requests.jsonl
```

Both inputs contain 216 total `work_units`. The fixed input spreads the work more evenly across the 8 shards, so the slowest shard finishes closer to the others.

Expected collector output shape:

```text
BATCH_INFERENCE_SUMMARY=.../outputs/bottleneck-job-array-imbalanced/run_summary.json
RECORDS_WRITTEN=32
SHARDS_COMPLETED=8
MAX_SHARD_ELAPSED=...
SHARD_IMBALANCE_RATIO=...
THROUGHPUT_BY_SLOWEST_SHARD=...
```

The printed values are also saved in:

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

`shard_elapsed_imbalance_ratio` compares the slowest shard with the fastest shard. The throughput by slowest shard is the practical end-to-end rate because the batch is not done until the slowest shard finishes.

In a real workload, the same symptom can come from uneven document lengths, image sizes, token counts, or known-heavy records. The next action is to shard by estimated work, then rerun the same job-array comparison.

> [!WARNING]
> The slowest shard controls walltime. Balance the work distribution before increasing scale.

## Part IV: Scale Across Nodes

The second central question is:

> Does moving from one LUMI-G node to two nodes add enough useful throughput?

Run:

```bash
sbatch jobs/run_16gcd_two_node.sh
```

For the provided synthetic workload, this should usually show that the second node still adds useful throughput. The point is to learn how to read the 8-to-16 incremental efficiency, not to force a failure.

After the job finishes:

```bash
python scripts/compare_scaling.py
```

Expected 16-GCD output shape:

```text
RUN_SUMMARY=.../outputs/synthetic-16gcd-two-node/run_summary.json
WORLD_SIZE=16
NODES=2
RANKS=16
TOTAL_THROUGHPUT=...
RANK_ELAPSED_SPREAD=...
```

The comparison uses the completed 1-GCD and 8-GCD runs from Part I and Part II.

Expected comparison output shape:

```text
| Configuration | World Size | Nodes | Throughput | Speedup | Efficiency | Diagnosis |
...
INCREMENTAL_SPEEDUP=...
INCREMENTAL_EFFICIENCY=...
```

Key signal:

```text
incremental_speedup_8_to_16 = throughput_16gcd / throughput_8gcd
incremental_efficiency_8_to_16 = incremental_speedup_8_to_16 / 2
```

For synchronized training, the job is only as fast as the slowest rank. Large rank elapsed spread means the faster ranks are waiting for the slowest one.

Interpretation:

| Observation | What to do | Reason |
|---|---|---|
| high 8-to-16 incremental efficiency | keep the two-node result | the second node adds useful throughput |
| low 8-to-16 incremental efficiency | stay at one node for this workload | the second node adds little value |
| valid 1-to-8 but poor 8-to-16 | inspect inter-node communication and synchronization | communication or synchronization may dominate |

A workload can scale well from 1 to 8 and still fail from 8 to 16. The 8-to-16 incremental efficiency is the key multi-node signal.

If multi-node scaling is poor, stay on one node for this workload until you have more useful compute per synchronization or less communication overhead.

### Optional Challenge: Communication-Bound Multi-Node Scaling

The healthy synthetic run above has enough compute per synchronization. This optional challenge makes communication more expensive relative to compute, then reduces that pressure.

Run the communication-heavy case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/comm_bound_8gcd.yaml \
  jobs/run_8gcd_single_node.sh
sbatch --export=ALL,CONFIG=configs/bottlenecks/comm_bound_16gcd.yaml \
  jobs/run_16gcd_two_node.sh
```

After both jobs finish:

```bash
python scripts/compare_scaling.py \
  --configs configs/bottlenecks/comm_bound_8gcd.yaml \
            configs/bottlenecks/comm_bound_16gcd.yaml
```

The demonstrated fix changes the synthetic workload from little compute and large repeated reductions:

```text
compute_repeats: 1
all_reduce_repeats: 6
all_reduce_elements: 4194304
```

to more compute and a small reduction:

```text
compute_repeats: 6
all_reduce_repeats: 1
all_reduce_elements: 2048
```

Run the improved case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/comm_improved_8gcd.yaml \
  jobs/run_8gcd_single_node.sh
sbatch --export=ALL,CONFIG=configs/bottlenecks/comm_improved_16gcd.yaml \
  jobs/run_16gcd_two_node.sh
```

After both jobs finish:

```bash
python scripts/compare_scaling.py \
  --configs configs/bottlenecks/comm_improved_8gcd.yaml \
            configs/bottlenecks/comm_improved_16gcd.yaml
```

If incremental efficiency improves, the bottleneck was communication relative to useful compute. In a real workload, the equivalent fix is usually to increase useful compute per synchronization, reduce synchronization frequency, or reduce communication volume.

## Repository Layout

```text
configs/    YAML configs for each run
examples/   JSONL inputs for bottleneck examples
jobs/       Slurm job scripts
logs/       Slurm output directory
scripts/    Workloads, collectors, and comparison helpers
```
