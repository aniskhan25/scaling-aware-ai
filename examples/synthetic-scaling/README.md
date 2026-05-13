# Synthetic Scaling Example

This example is the first complete runnable workflow in Scaling-Aware AI on LUMI.

It uses a controlled synthetic workload to compare:

- `synthetic-1gcd`
- `synthetic-8gcd-single-node`
- `synthetic-16gcd-two-node`

## Run

From the `scaling-aware-ai` directory:

```bash
sbatch jobs/run_1gcd.sh
sbatch jobs/run_8gcd_single_node.sh
sbatch jobs/run_16gcd_two_node.sh
```

After all jobs finish:

```bash
python scripts/compare_scaling.py
python scripts/validate_scaling_run.py
```

## Main Outputs

```text
outputs/scaling_report.md
outputs/scaling_report.json
outputs/synthetic-*/run_summary.json
outputs/synthetic-*/environment.json
outputs/synthetic-*/raw/placement_rank*.json
outputs/synthetic-*/raw/metrics_rank*.json
```

## What This Example Teaches

- how to run a staged scaling ladder
- how to collect placement metadata
- how to compare speedup and efficiency
- how to identify invalid scaling experiments
- why multi-node scaling should be justified by measured evidence
