# Scaling Report

## Run Set

- Workload:
- Date:
- Container:
- Git commit:
- Baseline config:
- Single-node config:
- Multi-node config:

## Results

| Configuration | World Size | Nodes | Throughput | Speedup | Efficiency | Decision |
|---|---:|---:|---:|---:|---:|---|
| 1 GCD | | | | 1.00 | 1.00 | baseline |
| 8 GCDs, 1 node | | | | | | |
| 16 GCDs, 2 nodes | | | | | | |

## Placement Checks

- Expected rank counts matched:
- Expected node counts matched:
- GPU-visible device counts looked correct:
- Per-rank throughput variance looked acceptable:

## Interpretation

State whether the result shows:

- good scaling
- poor but valid scaling
- invalid scaling

## Decision

Recommended next step:

- keep the smaller scale
- tune single-node behavior
- test a larger weak-scaling workload
- proceed to a larger staged run

## Notes

- Communication observations:
- Data/loading observations:
- Memory observations:
- Follow-up profiling needed:

