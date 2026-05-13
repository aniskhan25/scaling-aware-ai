# 6. Data Pipeline Scaling

Many AI jobs fail to scale because input delivery does not scale.

Before changing distributed launch settings, check whether the job is waiting for data, preprocessing, tokenization, decoding, or output writes.

## Synthetic Data Versus Real Data

Synthetic data is useful because it removes storage and preprocessing from the measurement.

Use synthetic data to ask:

- can the model compute path scale at all?
- is communication the obvious bottleneck?
- are ranks placed and launched correctly?

Use real data to ask:

- can the end-to-end workload scale?
- does input delivery keep up?
- does preprocessing dominate?
- do output artifacts create a bottleneck?

If synthetic data scales well but real data does not, the problem is probably not the GPU compute path.

## What To Measure

For training:

- data wait fraction
- samples/sec
- step time
- per-rank throughput variance
- CPU utilization
- GPU idle gaps
- checkpoint time

For batch inference:

- records/sec
- shard elapsed time
- failed shard count
- output write time
- model load time
- max shard elapsed time

## Common Data Bottlenecks

### Many Small Files

Many small files can create metadata and open/close overhead.

Prefer packed or sharded formats when possible:

- tar shards
- HDF5
- LMDB
- SquashFS-style packaged datasets
- larger JSONL shards for text workloads

### Duplicate Reads

Distributed jobs should avoid every rank reading the same records unless the workload explicitly requires it.

For training, use a distributed sampler or explicit rank sharding.

For batch inference, shard records by array task or by assigned file list.

### CPU Preprocessing

Tokenization, image decoding, compression, filtering, and augmentation can dominate runtime.

Measure this separately from GPU compute when possible.

### Output Pressure

Independent batch jobs can overload storage if every task writes many tiny files.

Prefer:

- one output shard per task
- one summary per task
- a later merge step
- restartable shard naming

## How The Examples Expose This

The DDP training example records `data_wait_fraction`. The default is zero because the data is synthetic, but the config includes `synthetic_data_wait_seconds` so users can simulate input delay and observe how it affects throughput.

The batch inference example writes one output JSONL and one summary JSON per array shard. This models a restartable pattern for independent records.

## Practical Rule

Compare synthetic and real-data runs before blaming communication.
