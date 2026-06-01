# CUDA Runtime Shape

## Proposed Backend Shape

### Phase 1: Host-Scheduled CUDA Backend

Bring up a CUDA runtime named `host_schedule` first. It is analogous to the
current `host_build_graph` runtime, but the name should describe the CUDA
execution model directly: the host owns graph scheduling and enqueues CUDA
kernel work.

The CUDA host runtime exports the same C API that `ChipWorker` already loads:

- lifecycle: `create_device_context`, `simpler_init`, `finalize_device`
- memory: `device_malloc_ctx`, `device_free_ctx`, `copy_to_device_ctx`,
  `copy_from_device_ctx`
- callable lifecycle: `prepare_callable`, `run_prepared`,
  `unregister_callable`
- diagnostics and comm symbols, initially as no-op or not-supported stubs

The CUDA `DeviceRunner` maps these onto CUDA runtime or driver APIs:

- `cudaSetDevice` or Driver API primary-context retention during
  `simpler_init`
- `cudaMalloc` / `cudaFree` or `cuMemAlloc` / `cuMemFree`
- `cudaMemcpyAsync` / `cuMemcpy*Async` on a runner-owned stream
- `cudaStreamSynchronize` or event synchronization at `run_prepared` boundary
- module loading through Driver API `cuModuleLoadDataEx` or CUDA runtime
  dynamic loading when the minimum toolkit supports it

`prepare_callable` compiles or loads user CUDA device code and caches module
state by `callable_id`. `run_prepared` launches one or more CUDA kernels from
host according to the prepared host graph.

This phase intentionally avoids device-side graph construction. It gives the
project a small compatibility slice:

- Python `Worker` / `ChipWorker` remains unchanged.
- `TaskArgs` and `CallConfig` stay the ABI boundary.
- Tensor memory allocation/copy paths become testable on a real CUDA device.
- User kernels can be expressed as CUDA global kernels before optimizing for
  persistent device-side scheduling.

### Phase 2: Persistent Device Runtime

Porting `tensormap_and_ringbuffer` requires a CUDA-native scheduling model.
The CUDA runtime should be named `persistent_device`. The recommended shape is
a persistent scheduler/worker kernel:

1. Host builds or uploads a runtime descriptor in global memory.
2. Host launches a persistent CUDA kernel with a configured grid.
3. Blocks or warps act as scheduler and worker roles inside that one kernel.
4. Workers call linked task functions through a generated dispatch table
   instead of launching arbitrary global kernels from device code.
5. Completion, dependency counters, and ready queues live in global memory.

This is closer to the existing AICPU/AICore design than host scheduling, but
the dispatch target changes. The scheduler must be part of the CUDA device
binary, and user compute must be compiled into task functions callable by the
persistent executor. See [persistent-device.md](../persistent-device.md) for the
static `nvcc` linking design and alternatives.

The first implementation should prefer normal offline `nvcc` compilation and
device linking. NVRTC plus nvJitLink can remain a later optimization path for
autotuning or shape-specialized runtime compilation, but they should not be the
foundation for the initial persistent runtime.
