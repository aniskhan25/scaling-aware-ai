# Batch Inference Job-Array Example

This example shows a workload where distributed training tools are unnecessary.

Each input record can be processed independently, so the guide uses a Slurm job array. Each array task handles a shard of the JSONL input and writes independent output and summary files.

## Run

From the `scaling-aware-ai` directory, edit the account line in the job script, then submit:

```bash
sbatch jobs/run_batch_inference_array.sh
```

After the array finishes, aggregate shard summaries:

```bash
python scripts/collect_batch_inference.py --config configs/batch-inference/job_array.yaml
```

## Outputs

```text
outputs/batch-inference-array/
  raw/
    outputs_shard0.jsonl
    summary_shard0.json
    ...
  run_summary.json
```

## What To Measure

Use inference-oriented metrics:

- records written
- records/sec
- shard elapsed time
- max shard elapsed time
- failed or missing shards

Do not report training samples/sec for this pattern. The unit of useful work is records processed.

## When This Pattern Is Appropriate

Use job arrays when:

- records are independent
- no collective communication is needed
- failures should be retryable per shard
- output files can be merged after the run
- throughput matters more than synchronous model updates

Use distributed inference only when model size, serving design, or cross-rank batching genuinely requires it.

