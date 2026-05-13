# 8. Bottleneck Demonstrations

This chapter turns scaling problems into small demonstrations that users can run and inspect.

Each demonstration follows the same pattern:

```text
Concept -> induced challenge -> observation -> resolution -> lesson
```

The point is not to manufacture impressive numbers. The point is to make bottlenecks visible enough that users can recognize the same pattern in their own workloads.

The code for these demonstrations is deliberately small. Each added knob exists to expose one observable bottleneck:

- `synthetic_data_wait_seconds` exposes input starvation in training.
- `work_units` exposes shard imbalance in independent batch work.
- workload shape settings expose too-small workloads and communication overhead.

## Demonstration 1: Data Starvation In Synchronized Training

### Concept

Distributed training only helps when each rank has enough ready work to keep the GPU busy. If every step waits for data, adding more GCDs can make the job larger without making it proportionally more useful.

### Induced Challenge

The DDP training script supports `synthetic_data_wait_seconds`. This simulates preprocessing or input delivery delay before each training step.

Bottleneck config:

```text
configs/bottlenecks/ddp_data_wait_bottleneck.yaml
```

Resolution config:

```text
configs/bottlenecks/ddp_data_wait_reduced.yaml
```

The first config injects a larger wait before every step. The second reduces that wait to model the effect of fixing input delivery.

### Run

Bottleneck case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_bottleneck.yaml \
  jobs/run_ddp_8gcd_config.sh
```

Resolution case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/ddp_data_wait_reduced.yaml \
  jobs/run_ddp_8gcd_config.sh
```

### Observe

Read:

```text
outputs/bottleneck-ddp-data-wait/run_summary.json
outputs/bottleneck-ddp-data-wait/raw/metrics_rank*.json
outputs/solution-ddp-data-wait-reduced/run_summary.json
outputs/solution-ddp-data-wait-reduced/raw/metrics_rank*.json
```

Focus on:

- `total_throughput_samples_per_sec`
- `mean_rank_throughput_samples_per_sec`
- `data_wait_fraction`
- `rank_elapsed_spread_seconds`

### Resolution

In a real workload, reducing data wait may mean:

- using larger dataset shards
- reducing many-small-file access
- moving expensive preprocessing offline
- increasing or tuning dataloader workers
- caching reusable transformed data
- ensuring each rank reads its own shard

The demo uses a config change to isolate the idea. Real workloads need a data pipeline change.

### Lesson

If data wait is high, scale-up is the wrong first fix. Make the current scale feed the GPUs reliably before using more of LUMI-G.

## Demonstration 2: Job-Array Shard Imbalance

### Concept

Independent batch work does not need distributed collectives, but it still needs balanced shards. Overall wall-clock time is often limited by the slowest shard.

### Induced Challenge

The batch inference script supports a `work_units` field. The bottleneck input puts heavy records on indices that all map to the same array shard under modulo sharding.

Bottleneck config:

```text
configs/bottlenecks/job_array_imbalanced.yaml
```

Resolution config:

```text
configs/bottlenecks/job_array_balanced.yaml
```

The balanced input spreads heavy records across shards.

### Run

Bottleneck case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_imbalanced.yaml \
  jobs/run_batch_inference_array_config.sh
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_imbalanced.yaml
```

Resolution case:

```bash
sbatch --export=ALL,CONFIG=configs/bottlenecks/job_array_balanced.yaml \
  jobs/run_batch_inference_array_config.sh
python scripts/collect_batch_inference.py \
  --config configs/bottlenecks/job_array_balanced.yaml
```

### Observe

Read:

```text
outputs/bottleneck-job-array-imbalanced/run_summary.json
outputs/bottleneck-job-array-imbalanced/raw/summary_shard*.json
outputs/solution-job-array-balanced/run_summary.json
outputs/solution-job-array-balanced/raw/summary_shard*.json
```

Focus on:

- `max_shard_elapsed_seconds`
- `throughput_records_per_sec_by_max_elapsed`
- per-shard `work_units_total`
- per-shard `elapsed_seconds`

### Resolution

In a real workload, shard balancing may mean:

- sharding by estimated token count, image size, or document length
- spreading known heavy records across shards
- using more smaller shards than workers
- retrying failed shards independently
- writing one output and one summary per shard

### Lesson

Job arrays remove collective communication, but they do not remove the need for measurement. The bottleneck can move from network synchronization to shard imbalance.

## Demonstration 3: Too-Small Workloads And Poor Efficiency

### Concept

Small workloads often scale poorly because communication and launch overhead are too large compared with useful compute.

### Induced Challenge

Use the synthetic ladder with a smaller workload shape by editing:

```text
configs/synthetic/single_node.yaml
configs/synthetic/two_node.yaml
```

Reduce values such as:

- `samples_per_step`
- `hidden_size`
- `compute_repeats`

Then compare against the default configuration.

### Observe

Read:

```text
outputs/scaling_report.md
```

Focus on:

- speedup
- efficiency
- whether poor efficiency appears at 8 GCDs or only at 16 GCDs

### Resolution

Increase useful work per rank before adding nodes:

- larger batch or token count
- larger model or sequence length
- gradient accumulation where algorithmically valid
- fewer unnecessary synchronization points

### Lesson

Poor scaling is sometimes the correct result. It may mean the workload does not have enough work to amortize distributed overhead.

## How To Use These Demos In Practice

For each real workload, write the same record:

```text
Observed symptom:
Likely bottleneck:
Evidence:
Fix attempted:
Result after fix:
Decision:
```

Use [scale-decision-record.md](../templates/scale-decision-record.md) to keep the result actionable.
