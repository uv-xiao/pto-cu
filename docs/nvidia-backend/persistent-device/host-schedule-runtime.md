# CUDA Persistent Device Runtime Analysis: Host-Schedule Runtime

## Host-Schedule Runtime

The first CUDA runtime should be named `host_schedule`, not `host_build_graph`.
The current Ascend `host_build_graph` name describes graph construction. For
CUDA the essential behavior is broader: the host owns scheduling and uses CUDA
streams to enqueue task kernels.

### Stream Semantics

A CUDA stream is an ordered queue. Work issued to the same stream executes in
issue order. Work issued to different streams may overlap only when:

- the device supports concurrent kernel execution;
- the kernels do not consume all execution resources;
- there is no dependency from events, synchronization, default-stream barriers,
  blocking allocations, or other implicit synchronization;
- the host enqueues independent work before waiting on it.

Therefore `host_schedule` needs multiple non-blocking streams if it wants
concurrent kernel execution. A single stream is correct but serializes all
task kernels.

Recommended `host_schedule` design:

```text
host scheduler
  |
  | ready task queue
  v
stream pool
  |  stream[0]  task A kernel -> event A
  |  stream[1]  task B kernel -> event B
  |  stream[2]  task C kernel -> event C
  |
  | dependencies represented by cudaEventRecord/cudaStreamWaitEvent
  v
worker.run returns after synchronizing all streams touched by this run
```

Rules:

- Create streams with `cudaStreamNonBlocking` or equivalent Driver API flags.
- Do not use the legacy default stream for task kernels.
- Carry the selected stream in the prepared callable manifest. The current
  bring-up ABI uses `PtoCudaHostCallable.version == 2` with `stream_id`, while
  version 1 callables remain mapped to stream 0.
- Track one event per task completion or per stream tail.
- Use `cudaStreamWaitEvent` to express dependencies between streams.
- Delay host synchronization until the existing `run_prepared` boundary unless
  a user-facing API requires earlier completion.
- Size the stream pool separately from `block_dim`; `block_dim` is a kernel
  launch-policy hint, not a stream count.
- Use `PTO_CUDA_STREAM_POOL_SIZE` to enlarge or shrink the host runtime stream
  pool for concurrency experiments. The default is 4 streams; invalid,
  zero, or overly large values fall back to the default.

`host_schedule` can provide concurrency for independent ready tasks, but it is
still host-dispatched. It pays host launch overhead per task and cannot match
the device-side dispatch latency target of `persistent_device`.

