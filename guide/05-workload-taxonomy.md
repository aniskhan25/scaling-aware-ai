# 5. Workload Taxonomy

Different AI workloads need different scaling patterns. A common mistake is to treat every large workload as a distributed training problem.

Start by classifying the workload.

## Synchronized Training

Use synchronized distributed training when multiple ranks cooperate on one model update stream.

Examples:

- DDP training
- full fine-tuning
- some LoRA or adapter training runs
- larger pretraining-style experiments

Scaling signals:

- samples/sec or tokens/sec
- step time
- global batch size
- gradient synchronization cost
- per-rank throughput variance
- checkpoint time

Main risks:

- small per-rank batches
- all-reduce overhead
- data loader stalls
- checkpoint bottlenecks
- poor rank placement

Use the [DDP training example](../examples/ddp-training/README.md) for a compact runnable pattern.

## Independent Batch Processing

Use job arrays or independent workers when each record can be processed without communicating with other records.

Examples:

- batch inference over documents
- embedding a corpus
- scoring evaluation cases
- synthetic data generation
- post-processing model outputs

Scaling signals:

- records/sec
- documents/sec
- tokens/sec
- shard completion count
- max shard elapsed time
- failed shard count

Main risks:

- many-small-file output patterns
- repeated model load overhead
- uneven shards
- missing or duplicated records
- output merge failures

Use the [batch inference job-array example](../examples/batch-inference/README.md) for a compact runnable pattern.

## Online Serving

Online serving optimizes for request latency and sustained throughput under realistic traffic.

Examples:

- API-backed model serving
- interactive assistants
- low-latency embedding services

Scaling signals:

- requests/sec
- p50/p95/p99 latency
- time to first token
- queue length
- batch fill rate
- error rate

Main risks:

- optimizing throughput while breaking latency
- poor batching policy
- KV-cache pressure
- load imbalance across replicas

Do not use the synthetic scaling ladder as a serving benchmark. It does not model queueing, request arrival patterns, or latency percentiles.

## Large Model Memory Scaling

Some workloads scale because the model or optimizer state does not fit on one visible device.

Examples:

- FSDP
- DeepSpeed ZeRO
- tensor parallel inference
- pipeline or tensor parallel training

Scaling signals:

- memory per rank
- activation memory
- optimizer state memory
- tokens/sec
- checkpoint time
- communication/computation overlap

Main risks:

- sharding overhead
- complicated checkpointing
- poor overlap
- communication-heavy layers

This guide's first passes focus on DDP and job arrays. Memory-sharded patterns belong in a later framework-specific expansion.

## Decision Table

| Workload | Preferred First Pattern | Main Metric | Avoid |
|---|---|---|---|
| One model, synchronized updates | DDP single node | samples/sec or tokens/sec | job arrays |
| Independent records | Slurm job array | records/sec | DDP |
| Large model does not fit | sharding or model parallelism | tokens/sec and memory/rank | naive DDP |
| Online service | replicas and batching | latency percentiles | synthetic throughput-only tests |
| Data preprocessing | job arrays or staged pipeline | records/sec and failures | multi-node collectives |

## Practical Rule

Choose the parallel pattern that matches the dependency structure of the work.

If records do not need to communicate, do not introduce distributed collectives.

