# NVIDIA Backend Flow Details: User-Facing Compile and Launch Flow

## User-Facing Compile and Launch Flow

### Current a2a3/a5 Flow

```text
Python scene/example
  |
  | RuntimeBuilder(platform).get_binaries(runtime, build?)
  v
build/lib/{arch}/{variant}/{runtime}/
  |  libhost_runtime.so
  |  libaicpu_kernel.so
  |  aicore_kernel.o or libaicore_kernel.so for sim
  |
  | KernelCompiler(platform)
  |  - compile_orchestration(runtime, orch.cpp) -> orchestration .so bytes
  |  - compile_incore(kernel.cpp, core_type) -> AICore/AIV bytes
  v
ChipCallable bytes
  |
  | Worker(level=2).register(chip_callable) -> callable_id
  | Worker.init()
  v
ChipWorker.init(device_id, RuntimeBinaries)
  |  dlopen(libhost_runtime.so)
  |  load aicpu/aicore binary files into host memory
  |  create DeviceRunner context
  |  simpler_init(ctx, device_id, log config)
  |
  | Worker.run(callable_id, TaskArgs, CallConfig)
  v
ChipWorker.run_prepared()
  |  view_to_chip_storage(args)
  |  run_prepared C API
  v
host runtime DeviceRunner
  |  prepare runtime args
  |  upload/reuse user kernel and orchestration payloads
  |  launch AICPU scheduler + AICore executor
  |  synchronize stream
  v
results copied back by runtime or explicit worker.copy_from()
```

The important property is that runtime binaries and user callables are
different artifacts:

- runtime binaries are platform/runtime infrastructure built by
  `RuntimeBuilder`;
- `ChipCallable` is user work compiled by `KernelCompiler`;
- `ChipWorker` binds them at runtime through the stable host-runtime C API.

`host_build_graph` and `tensormap_and_ringbuffer` differ after `run_prepared`
enters the host runtime. The outer Python flow stays the same. CUDA should use
runtime names that describe its own execution model: `host_schedule` and
`persistent_device`.

### Proposed CUDA Flow

```text
Python scene/example
  |
  | RuntimeBuilder(platform="cuda").get_binaries(runtime, build?)
  v
build/lib/cuda/onboard/{runtime}/
  |  libhost_runtime.so
  |  optional executor fatbin/cubin/PTX
  |
  | CudaKernelCompiler or extended KernelCompiler
  |  - compile host_schedule callable -> fatbin/cubin/PTX + manifest
  |  - optional compile orchestration metadata for host scheduler
  v
ChipCallable-compatible CUDA payload
  |
  | Worker(level=2, platform="cuda").register(payload) -> callable_id
  | Worker.init()
  v
ChipWorker.init(device_id, RuntimeBinaries)
  |  dlopen(CUDA libhost_runtime.so)
  |  create CUDA DeviceRunner context
  |  retain or create CUDA context for device_id
  |  create stream(s)
  |
  | Worker.run(callable_id, TaskArgs, CallConfig)
  v
ChipWorker.run_prepared()
  |  same C API call shape as a2a3/a5
  v
CUDA host runtime
  |  prepare_callable: load module/library and cache function handles
  |  run_prepared: build kernel params and launch CUDA kernel(s)
  |  synchronize stream or event at the existing run boundary
  v
results copied back by runtime or explicit worker.copy_from()
```

The CUDA target should preserve the `simpler` usage model: users still compile
a callable, register it, initialize a `Worker`, and call `run`. The backend
changes what the callable payload contains and how the host runtime launches
it.

