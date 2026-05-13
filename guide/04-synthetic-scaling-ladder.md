# 4. Scaling Decision Ladder

This chapter is the heart of the guide.

The question is not:

> Can I run on more GPUs?

The useful question is:

> What observation at the current scale proves that the next scale is the right next experiment?

The synthetic scripts in this guide are a controlled way to practice that reasoning. They are not presented as a benchmark of LUMI or a realistic model. They exist so you can observe what changes when you move from one visible device, to one full LUMI-G node, to multiple nodes.

## Stage 0: Define The Workload Before Scaling

Before launching anything, define the unit of useful work.

Examples:

- training: samples/sec, tokens/sec, step time, validation progress
- batch inference: records/sec, tokens/sec, completed outputs
- online inference: latency percentiles, requests/sec, time to first token
- embeddings: documents/sec, chunks/sec, indexable vectors/sec

If the useful work unit is not defined, scaling results will be misleading. "The job completed" is not a scaling metric.

Also decide whether the workload is:

- synchronized, where ranks must communicate
- independent, where records can be processed separately
- memory-limited, where scale is needed because the model does not fit
- latency-limited, where bigger batches may hurt the actual objective

Do not move up the ladder until this is clear.

## Stage 1: Single-GCD Baseline

The single-GCD baseline answers:

> Is the smallest meaningful GPU run healthy enough to deserve scaling?

Collect:

- steady-state throughput
- elapsed time after warmup
- GPU-visible device count
- memory behavior
- data wait or input delay
- environment and container metadata
- a placement file for rank 0

Healthy observations:

- throughput is repeatable across runs
- warmup is separated from steady state
- memory is stable
- output is correct
- the useful metric matches the workload objective

Warning observations:

- first iterations dominate the timing
- GPU is often idle because data is late
- memory is near failure before any scale-up
- throughput varies heavily between runs
- the code only works with a tiny debug input

Decision:

| Observation | Meaning | Next Action |
|---|---|---|
| stable throughput, enough work per step | baseline is worth scaling | try full-node |
| poor throughput with synthetic data | compute/model path is weak | profile or optimize single-GCD first |
| good synthetic throughput, poor real-data throughput | data path is likely limiting | fix input pipeline before scaling |
| memory already near limit | scale may be about memory, not speed | consider sharding strategy instead of plain DDP |
| correctness not stable | performance is irrelevant | fix correctness first |

Rule:

Do not use multiple nodes to hide a bad single-GCD baseline.

## Stage 2: Full-Node Test

The full-node test answers:

> Does using all 8 GPU-visible devices on one LUMI-G node improve useful throughput enough to justify distributed execution?

This step introduces:

- multiple ranks
- rank-to-device mapping
- CPU affinity effects
- intra-node collective communication
- per-rank imbalance

Collect:

- total throughput
- mean/min/max rank throughput
- speedup versus 1 GCD
- efficiency versus 1 GCD
- placement files for all ranks
- node count and world size

Interpretation:

| Observation | Likely Meaning | Next Action |
|---|---|---|
| speedup is strong and efficiency is acceptable | workload has enough per-rank work | consider multi-node only if walltime or throughput needs remain |
| throughput rises but efficiency is poor | larger run may be faster but wasteful | compare GPU-hour cost before proceeding |
| rank count or placement is wrong | experiment is invalid | fix launch before interpreting performance |
| one or two ranks are much slower | placement, CPU binding, or local contention issue | inspect per-rank placement and affinity |
| 8-GCD run barely beats 1 GCD | workload too small or synchronization-heavy | increase work per rank or stay smaller |

Important distinction:

Raw throughput usually improves when you add enough devices. That does not mean scaling is good. Efficiency tells you whether the improvement is proportional to the resources consumed.

Rule:

Do not move to multi-node until single-node behavior is understandable.

## Stage 3: Multi-Node Test

The multi-node test answers:

> After single-node scaling works, does adding the network still improve useful throughput?

This step introduces:

- inter-node communication
- rendezvous behavior
- network interface choices
- more failure modes
- more GPU-hour cost

Collect:

- 16-GCD throughput
- speedup versus 1 GCD
- incremental speedup versus 8 GCDs
- efficiency versus 1 GCD
- hostnames represented in placement files
- per-rank throughput spread

Interpretation:

| Observation | Likely Meaning | Next Action |
|---|---|---|
| 8-GCD and 16-GCD both scale well | workload may justify larger staged runs | increase scale gradually |
| 8-GCD good, 16-GCD poor | network or cross-node communication is limiting | inspect communication pattern and workload size |
| 16-GCD barely improves over 8-GCD | multi-node is not justified for this workload size | stay single-node or increase per-rank work |
| validation reports wrong node count | result is invalid | fix launch/rendezvous |
| per-rank variance grows on two nodes | imbalance or network placement may matter | inspect host/rank mapping |

Rule:

Multi-node is a second-order scaling decision. It should follow a successful single-node result, not replace it.

## Stage 4: Decide If This Should Be Distributed At All

Some workloads do not need distributed collectives.

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

Examples:

| Workload | Better First Pattern | Why |
|---|---|---|
| DDP training | full-node DDP | gradients synchronize each step |
| corpus embedding | job array or independent workers | documents are independent |
| batch evaluation | job array | cases can be scored separately |
| large model that does not fit | sharding/model parallelism | memory, not simple throughput, is the blocker |
| online serving | replicas and batching | latency and queueing matter |

Rule:

If the work is independent, distributed collectives are usually unnecessary complexity.

## Stage 5: Turn Measurements Into A Scale Decision

A scaling report should end with a decision, not just a table.

Use this structure:

```text
Workload objective:
Useful metric:
Baseline result:
Single-node result:
Multi-node result:
Observed bottleneck:
Chosen scale:
Why this scale:
What would justify moving up:
What should be fixed first:
```

Example interpretations:

### Case A: Good Single-Node, Poor Multi-Node

Observation:

- 8 GCDs give strong speedup
- 16 GCDs add little
- placement is valid

Interpretation:

The workload is probably large enough for one node but not large enough to amortize cross-node communication.

Decision:

Use one full node for now. Try multi-node again only after increasing per-rank work, reducing synchronization frequency, or changing the workload shape.

### Case B: Poor Single-Node

Observation:

- 8 GCDs give weak speedup
- per-rank throughput varies
- placement is valid

Interpretation:

The issue appears before the network is involved. Multi-node will likely make the problem harder to diagnose.

Decision:

Stay at one node or below. Inspect workload size, CPU binding, data loading, and communication frequency.

### Case C: Synthetic Scales, Real Data Does Not

Observation:

- synthetic ladder scales acceptably
- real workload stalls
- GPU idle gaps or high data wait appear

Interpretation:

The launch and compute path are probably not the first problem. The data path is limiting useful throughput.

Decision:

Fix file format, sharding, dataloader settings, preprocessing, or storage behavior before scaling further.

### Case D: Job Array Beats Distributed Launch

Observation:

- records are independent
- job-array shards complete cleanly
- no collective communication is needed

Interpretation:

The workload does not need distributed training machinery.

Decision:

Use job arrays or independent workers. Track records/sec, failed shards, and merge correctness.

## Runbook: Synthetic Ladder

Use this runbook after you understand what each stage is meant to prove.

Working directory:

```bash
cd /path/to/scaling-aware-ai
```

Submit:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
sbatch jobs/run_16gcd_two_node.sh
```

Compare and validate:

```bash
python scripts/compare_scaling.py
python scripts/validate_scaling_run.py
```

Read:

```text
outputs/scaling_report.md
outputs/synthetic-1gcd/run_summary.json
outputs/synthetic-8gcd-single-node/run_summary.json
outputs/synthetic-16gcd-two-node/run_summary.json
outputs/synthetic-*/raw/placement_rank*.json
```

## Practical Rule

Scaling is not a ladder you climb automatically. It is a ladder where each rung must justify the next one.
