# Product Brief

## Working Title

Scaling-Aware AI on LUMI: A Practical Guide to Efficient Multi-GPU and Multi-Node Workloads

## Product Type

Standalone GitHub-style technical guide.

The guide should be readable independently of the main LUMI AI Guide, while still being able to reuse or reference scripts, lessons, and examples from it during development.

## Problem

Many AI users can submit jobs on LUMI but do not yet have a reliable method for deciding whether a workload should use 1 GCD, a full node, multiple nodes, job arrays, or a different optimization path entirely.

Common problems include:

- scaling before the single-device baseline is healthy
- confusing raw throughput improvement with efficient scaling
- failing to validate rank placement and GPU visibility
- treating data-loading bottlenecks as distributed-training bottlenecks
- using multi-node jobs for workloads that would be better served by job arrays or single-node batching
- omitting GPU-hour cost from scale-up decisions

## Product Goal

Give LUMI users a repeatable engineering workflow for scaling AI workloads:

- measure correctly
- launch correctly
- diagnose bottlenecks correctly
- interpret scaling results correctly
- choose the smallest scale that satisfies the workload objective

## Non-Goals

The guide will not:

- attempt to benchmark all frameworks on LUMI
- promise universal scaling numbers
- replace official LUMI documentation
- teach basic Python, Slurm, or PyTorch from scratch
- optimize every possible model architecture
- encourage multi-node scaling before baseline validation

## Primary User Journeys

### User Journey 1: From Single GCD to Full Node

A user has a working single-GCD training or inference script and wants to know whether using all 8 visible devices on a LUMI-G node is worthwhile.

The guide should help them:

- record a clean baseline
- inspect placement
- run a single-node scaling test
- compare throughput and efficiency
- decide whether to tune, scale, or stop

### User Journey 2: From Full Node to Multi-Node

A user has acceptable single-node performance and wants to try 2 or more nodes.

The guide should help them:

- validate rendezvous and rank counts
- identify when inter-node communication dominates
- compare strong and weak scaling
- make a GPU-hour-aware decision

### User Journey 3: Diagnosing Poor Scaling

A user has a distributed job that launches but scales poorly.

The guide should help them:

- separate invalid launches from valid but poor scaling
- inspect rank placement and per-rank throughput
- test synthetic data versus real data
- decide whether the bottleneck is compute, communication, data, memory, or launch overhead

### User Journey 4: Choosing the Right Parallel Pattern

A user has a large AI workload but is unsure whether it needs distributed training, distributed inference, job arrays, or a staged single-node workflow.

The guide should help them:

- classify the workload
- choose a scaling pattern
- avoid unnecessary distributed complexity
- plan capacity and artifacts

## Success Criteria

The guide is successful if a LUMI user can:

- explain the difference between MI250X modules, GCDs, and software-visible devices
- collect placement metadata for every rank
- run a 1-GCD, 8-GCD, and multi-node scaling ladder
- compute speedup and scaling efficiency
- identify invalid scaling experiments
- decide when multi-node scaling is justified
- document a scale-up decision with measured evidence

## Editorial Position

The guide should be practical, direct, and evidence-driven.

Each technical recommendation should answer:

- what problem it solves
- why it matters on LUMI
- how to test it
- what result means success or failure
- what to try next

