# 3. Scaling Metrics

Scaling decisions should be based on a small set of consistent metrics.

## Throughput

Throughput measures useful work per second.

Examples:

- samples/sec
- tokens/sec
- images/sec
- documents/sec
- requests/sec

The synthetic ladder reports samples/sec. Real workloads should choose a unit that reflects useful work.

## Speedup

Speedup compares a larger run to the baseline:

```text
speedup = throughput_target / throughput_baseline
```

If 1 GCD produces 100 samples/sec and 8 GCDs produce 600 samples/sec, speedup is 6.0.

## Scaling Efficiency

Efficiency compares speedup to the increase in world size:

```text
efficiency = speedup / (target_world_size / baseline_world_size)
```

For a 1-GCD to 8-GCD comparison:

- 8x speedup gives 1.00 efficiency
- 6x speedup gives 0.75 efficiency
- 4x speedup gives 0.50 efficiency
- 2x speedup gives 0.25 efficiency

Efficiency matters because raw throughput can increase while resource use becomes unattractive.

## Strong Scaling

Strong scaling keeps the global workload fixed while adding devices.

It answers:

How much faster can I finish the same amount of work?

Strong scaling often shows lower efficiency as device count rises because each rank gets less work while communication remains significant.

## Weak Scaling

Weak scaling keeps per-device work roughly fixed while adding devices.

It answers:

Can I process a larger workload in roughly the same time?

Weak scaling is often more relevant for production throughput and large training runs.

## Per-Rank Variance

Per-rank metrics help identify imbalance. If one rank is much slower, the full job waits.

Watch:

- min rank throughput
- max rank throughput
- elapsed time spread
- per-rank placement differences

## Cost-Normalized Thinking

A larger run should be judged against GPU-hours, not throughput alone.

```text
gpu_hours = number_of_gcds * walltime_hours
```

Sometimes a less efficient larger run is still acceptable because wall-clock time matters. That decision should be explicit.

## Practical Rule

Report throughput, speedup, and efficiency together.

