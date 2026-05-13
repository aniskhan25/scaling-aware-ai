# Technical Scope

## Target Platform

The guide targets LUMI-G AI workloads.

Important platform facts to explain and validate in examples:

- LUMI-G exposes 8 GPU-visible devices per full node.
- Each full node contains 4 AMD MI250X modules.
- Each MI250X contains 2 GCDs.
- Jobs can use 56 CPU cores per LUMI-G node.
- Rank placement, CPU binding, and communication topology can affect observed performance.

## Initial Framework Scope

The initial implementation should focus on PyTorch first because it is the most direct continuation of the current AI Guide material.

Initial framework coverage:

- PyTorch
- `torch.distributed`
- DDP-style launch patterns
- synthetic workload scripts
- simple model-training example
- job-array batch inference pattern
- bottleneck demonstrations for data wait and shard imbalance

Planned later coverage:

- DeepSpeed
- FSDP
- Megatron-style parallelism
- inference frameworks
- embedding workloads

## Required Runnable Artifacts

The guide should eventually include runnable artifacts in these categories.

### Jobs

```text
jobs/
  run_1gcd.sh
  run_8gcd_single_node.sh
  run_16gcd_two_node.sh
  run_job_array_inference.sh
```

### Scripts

```text
scripts/
  inspect_placement.py
  run_synthetic_workload.py
  collect_metrics.py
  compare_scaling.py
  validate_scaling_run.py
  summarize_environment.py
```

### Configs

```text
configs/
  synthetic/
    baseline.yaml
    single_node.yaml
    two_node.yaml
```

### Templates

```text
templates/
  run-manifest.yaml
  scaling-report.md
  capacity-plan.md
  post-run-review.md
```

### Examples

```text
examples/
  synthetic-scaling/
  ddp-training/
  transformer-finetuning/
  batch-inference/
```

## Measurement Scope

Every serious example should record:

- world size
- node count
- rank and local rank
- hostname
- visible GPU count
- selected device
- elapsed time
- throughput
- per-rank throughput
- total throughput
- speedup versus baseline
- efficiency versus baseline
- relevant environment variables
- Slurm job metadata

Optional measurements:

- GPU utilization
- GPU memory
- CPU utilization
- dataloader wait time
- checkpoint time
- profiler traces

## Decision Rules

The guide should consistently apply these rules:

1. Do not scale before the single-GCD run is stable.
2. Do not interpret throughput before validating placement.
3. Do not compare runs with different workload definitions unless the scaling mode is explicitly weak scaling.
4. Do not use multi-node runs when job arrays or single-node batching solve the problem more simply.
5. Do not judge success from raw throughput alone; include efficiency and GPU-hour cost.

## Integration With Existing AI Guide Material

The standalone guide can reuse technical ideas from:

- `extension-track/06-topology-aware-scaling`
- `extension-track/12-cost-awareness-and-capacity-planning`
- `5-multi-gpu-and-node`
- `6-monitoring-and-profiling`
- `3-file-formats`

However, the standalone guide should not assume users are following the original lesson order. Any dependency should be stated directly or linked as optional background.

## External References To Keep

The guide should link to official LUMI resources for platform details:

- LUMI-G hardware overview
- LUMI network and interconnect documentation
- LUMI job distribution and binding documentation
- LUMI-G batch script examples
- LUMI training materials on architecture and profiling

## Current Implementation Scope

The current implementation includes:

- synthetic scaling example
- placement inspection
- 1-GCD, 8-GCD, and 16-GCD launch scripts
- comparison report
- validation command

These files live in:

- `guide/04-synthetic-scaling-ladder.md`
- `jobs/`
- `scripts/`
- `configs/synthetic/`
- `examples/synthetic-scaling/`

The current workload-specific examples live in:

- `guide/05-workload-taxonomy.md`
- `guide/06-data-pipeline-scaling.md`
- `guide/07-workload-specific-examples.md`
- `configs/ddp-training/`
- `configs/batch-inference/`
- `jobs/run_ddp_*.sh`
- `jobs/run_batch_inference_array.sh`
- `scripts/run_ddp_training.py`
- `scripts/run_batch_inference.py`
- `scripts/collect_batch_inference.py`
- `examples/ddp-training/`
- `examples/batch-inference/`

The bottleneck demonstration material lives in:

- `guide/08-bottleneck-demonstrations.md`
- `configs/bottlenecks/`
- `examples/bottlenecks/`
- `jobs/run_ddp_8gcd_config.sh`
- `jobs/run_batch_inference_array_config.sh`
