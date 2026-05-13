# Content Architecture

## Guide Structure

The long-form guide should be organized as a practical manual, not a course sequence. Users should be able to read it end to end or jump directly to a diagnosis section.

Recommended top-level structure:

```text
guide/
  01-introduction.md
  02-lumi-g-mental-model.md
  03-scaling-metrics.md
  04-single-gcd-baseline.md
  05-workload-taxonomy.md
  06-data-pipeline-scaling.md
  07-workload-specific-examples.md
  08-bottleneck-demonstrations.md
  09-launch-patterns.md
  10-placement-and-binding.md
  11-scaling-ladders.md
  12-communication-bottlenecks.md
  13-profiling-and-observability.md
  14-interpreting-results.md
  15-optimization-playbooks.md
  16-cost-and-capacity.md
  17-reproducibility.md
  18-troubleshooting.md
```

## Chapter Template

Each chapter should use this pattern:

```text
Problem
Why it matters on LUMI
Mental model
Minimal practical example
What to measure
How to interpret results
Common failure modes
Recommendations
Checklist
```

## Chapter Summaries

### 1. Introduction

Explain why scaling-aware AI matters on LUMI. Establish the central rule: scaling is a measured decision, not simply using more devices.

### 2. LUMI-G Mental Model

Explain the software-visible LUMI-G topology:

- 4 AMD MI250X modules per node
- 2 GCDs per MI250X
- 8 GPU-visible devices per full node
- 56 CPU cores available to jobs
- CPU NUMA considerations
- Slingshot network behavior for multi-node jobs

### 3. Scaling Metrics

Define the metrics used throughout the guide:

- throughput
- latency
- speedup
- scaling efficiency
- GPU-hour-normalized throughput
- strong scaling
- weak scaling
- per-rank variance

### 4. Single-GCD Baseline

Show how to produce a trustworthy baseline before scaling:

- container and package recording
- GPU visibility check
- warmup handling
- steady-state measurement
- data-loading sanity checks
- memory usage

### 5. Workload Taxonomy

Map workload types to scaling approaches:

- data-parallel training
- large-model fine-tuning
- pretraining
- batch inference
- online serving
- embeddings and RAG
- synthetic data
- evaluation workloads

### 6. Data Pipeline Scaling

Show how input delivery limits scaling:

- many small files
- packed formats
- sharding
- dataloader workers
- CPU preprocessing
- storage pressure
- synthetic-data controls

### 7. Workload-Specific Examples

Compare representative patterns:

- synchronized DDP training
- independent batch inference with job arrays
- when distributed collectives are unnecessary

### 8. Bottleneck Demonstrations

Show concept, challenge, observation, and resolution for:

- data starvation in synchronized training
- shard imbalance in job arrays
- too-small workloads with poor scaling efficiency

### 9. Launch Patterns

Provide LUMI job launch patterns:

- single-GCD debug
- full-node 8-GCD
- two-node 16-GCD
- job-array pattern for embarrassingly parallel workloads

### 10. Placement and Binding

Teach users how to record and inspect:

- rank
- local rank
- hostname
- CPU affinity
- visible device count
- selected device
- Slurm metadata
- environment variables

### 11. Scaling Ladders

Explain how to design controlled scaling experiments:

- 1 GCD to 8 GCDs to 16 GCDs
- strong scaling
- weak scaling
- one-variable-at-a-time experiments
- valid versus invalid comparisons

### 12. Communication Bottlenecks

Explain distributed communication:

- all-reduce
- all-gather
- reduce-scatter
- broadcast
- communication/computation ratio
- single-node versus multi-node differences

### 13. Profiling and Observability

Define profiling levels:

- lightweight metrics for every run
- framework profiling for suspicious stages
- ROCm/system profiling for deeper diagnosis

### 14. Interpreting Results

Convert metrics into decisions:

- good single-node and multi-node scaling
- poor single-node scaling
- good single-node but poor multi-node scaling
- higher throughput but poor efficiency
- inconsistent or invalid results

### 15. Optimization Playbooks

Give symptom-driven recommendations:

- low single-GCD throughput
- poor 8-GCD scaling
- poor multi-node scaling
- dataloader stalls
- checkpoint bottlenecks
- rank imbalance
- high utilization but poor useful throughput

### 16. Cost and Capacity

Connect scaling to GPU-hour use:

- GPU-hours
- walltime
- queue behavior
- staged runs
- stop rules
- production run planning

### 17. Reproducibility

Define required run artifacts:

- manifest
- config
- environment
- Slurm metadata
- placement metadata
- metrics
- logs
- summaries

### 18. Troubleshooting

Build a catalog of common failures:

- distributed launch hangs
- wrong world size
- missing GPU visibility
- duplicated output writes
- rank imbalance
- poor multi-node scaling
- inconsistent measurements

## Navigation Requirements

The guide should support:

- end-to-end reading
- quick diagnosis from symptoms
- copyable job templates
- direct links from concepts to scripts
- concise checklists at the end of chapters

## Style Rules

- Use concrete LUMI examples.
- Prefer measured decision rules over vague tuning advice.
- Make every command explain what it validates.
- Treat multi-node scaling as something to justify, not a default goal.
- Keep examples runnable and artifact-producing.
