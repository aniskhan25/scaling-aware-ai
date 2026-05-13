# 1. Introduction

Scaling-aware AI means treating scale as an engineering decision, not a default setting.

On LUMI-G, adding more GPU-visible devices changes more than device count. It changes rank placement, CPU affinity, intra-node communication, network behavior, memory pressure, data-loading pressure, queue cost, and GPU-hour consumption. A larger job can finish faster and still be a poor use of resources if scaling efficiency collapses.

This guide starts from one practical rule:

Do not scale an unstable or undersized workload.

## The Scaling-Aware Workflow

Use this workflow for any serious scale-up decision:

1. Establish a clean 1-GCD baseline.
2. Record environment and placement metadata.
3. Run an 8-GCD single-node comparison.
4. Run a 16-GCD two-node comparison only after the single-node result is interpretable.
5. Compare throughput, speedup, efficiency, and placement validity.
6. Decide whether to scale, tune, simplify, or stop.

## What Counts As A Useful Scaling Result

A useful result is not just a faster run. It is a run where:

- rank counts match the intended launch
- node counts match the intended launch
- placement metadata exists for every rank
- throughput is measured after warmup
- speedup is compared against a baseline
- efficiency is interpreted against the added device count
- the next action is explicit

## First Runnable Slice

The first runnable example in this guide is a synthetic scaling ladder:

- 1 GCD
- 8 GCDs on one node
- 16 GCDs on two nodes

The synthetic workload is intentionally simple. It performs repeated dense matrix operations and a distributed all-reduce. It is not a model-quality benchmark. Its purpose is to teach the decision process: what to measure at each scale, what the result means, and whether the next larger run is justified.

## What To Read Next

- Read [LUMI-G mental model](02-lumi-g-mental-model.md) before interpreting topology effects.
- Read [scaling metrics](03-scaling-metrics.md) before comparing results.
- Follow [scaling decision ladder](04-synthetic-scaling-ladder.md) to understand when to move up, stop, or fix the current scale.
