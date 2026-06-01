# CUDA Persistent Device Runtime Analysis: Orchestrator Separation

## Orchestrator Separation

The hardest part is the orchestrator. In today's Ascend runtime, orchestration
can be a separately compiled payload. For CUDA `persistent_device`, anything
that runs on the GPU must be compiled into the final device module before the
persistent executor is launched.

There are three viable models.

### Model A: Host Orchestrator, Device Scheduler

The host builds task descriptors and dependencies, then the persistent executor
only schedules ready work on device.

Pros:

- keeps existing orchestration `.so` style closer to today;
- easier first persistent executor;
- avoids device compiling user orchestration code.

Cons:

- not equivalent to `tensormap_and_ringbuffer` if the goal is device-side graph
  construction;
- host still pays graph-build cost;
- dynamic dependencies discovered on device are not supported.

### Model B: Device Orchestrator Linked Into Executor

The user orchestration code is CUDA device code. The callable compiler links
it into the same final module as the scheduler, worker runner, task bodies,
and generated dispatch.

Pros:

- closest CUDA analogue to device-side `tensormap_and_ringbuffer`;
- all graph build and scheduling state can live in device global memory;
- no device-side dynamic linking needed after launch.

Cons:

- orchestration source must obey CUDA device-code restrictions;
- per-callable device link is mandatory;
- Python/C++ codegen must generate a device entry and manifest;
- debugging is harder than host orchestration.

### Model C: Host Submits Descriptors Incrementally

The host runs orchestration and pushes task descriptors into a device queue
while a persistent scheduler is already running.

Pros:

- keeps user orchestration on host;
- scheduler/worker latency after submit is device-side;
- supports streaming workloads.

Cons:

- host/device queue protocol is more complex;
- correctness depends on host/device memory ordering and queue back-pressure;
- still not full device-side graph construction.

Recommended order:

1. `host_schedule` for correct CUDA execution and stream concurrency.
2. `persistent_device` Model A or C as a tracer bullet for device worker
   dispatch.
3. `persistent_device` Model B only after the task ABI and scheduler queues are
   stable.

