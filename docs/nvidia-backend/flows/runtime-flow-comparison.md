# NVIDIA Backend Flow Details: Runtime Flow Comparison

## Runtime Flow Comparison

### a2a3/a5 Hardware Flow

```text
install/build time:
  pip install . or --build
    -> build_runtimes.py
    -> RuntimeBuilder
    -> RuntimeCompiler
    -> host.so + aicpu.so + aicore.o

per callable:
  KernelCompiler.compile_orchestration()
    -> orchestration .so bytes
  KernelCompiler.compile_incore()
    -> kernel object/text bytes per func_id
  ChipCallable.build()
    -> orch bytes + child CoreCallable bytes

Worker.init:
  Python ChipWorker wrapper
    -> C++ ChipWorker::init
    -> load libsimpler_log.so globally
    -> dlopen host runtime locally
    -> dlsym full pto_runtime_c_api.h surface
    -> create_device_context()
    -> read aicpu/aicore binaries into host buffers
    -> simpler_init(ctx, device_id, log level)
    -> DeviceRunner attaches current thread to Ascend device

Worker.prepare/register:
  L2 direct path:
    -> Worker.register returns cid
    -> first run prepares lazily, or explicit prepare_callable
  L3 child process path:
    -> parent prewarms child via mailbox _CTRL_PREPARE
    -> child calls ChipWorker.prepare_callable(cid, callable)

prepare_callable in host runtime:
  -> upload child kernel binaries into device memory or sim cache
  -> upload orchestration SO payload
  -> cache prepared state keyed by callable_id/build-id
  -> AICPU or host side can dlopen orchestration once per cid

run_prepared in host runtime:
  -> copy TaskArgs into runtime ABI storage
  -> allocate/copy tensors as needed
  -> build or bind runtime graph
  -> copy Runtime descriptor to device
  -> launch AICPU init/main scheduler
  -> launch AICore executor
  -> scheduler dispatches tasks to AICore workers
  -> synchronize stream
  -> copy outputs and diagnostic data back

finalize:
  -> unregister callable state when requested
  -> free device allocations and streams
  -> destroy context
  -> dlclose host runtime
```

### CUDA Host-Schedule Flow

```text
install/build time:
  pip install . or --build
    -> build CUDA host runtime
    -> optionally build host_schedule support module/fatbin

per callable:
  CUDA callable compiler
    -> CUDA source/device IR -> PTX, cubin, or fatbin
    -> manifest: entry names, argument layout, launch policy
  ChipCallable-compatible payload
    -> module image bytes + metadata

Worker.init:
  Python ChipWorker wrapper
    -> C++ ChipWorker::init
    -> load libsimpler_log.so globally
    -> dlopen CUDA host runtime locally
    -> dlsym same pto_runtime_c_api.h surface
    -> create_device_context()
    -> simpler_init(ctx, device_id, log level)
    -> CUDA DeviceRunner retains/sets current context
    -> CUDA DeviceRunner creates streams/events

prepare_callable in CUDA host runtime:
  -> load module/library from payload bytes
  -> get function/kernel handles by entry name
  -> cache handles and manifest by callable_id
  -> optionally set per-kernel attributes

run_prepared in CUDA host runtime:
  -> translate ChipStorageTaskArgs into CUDA kernel parameter arrays
  -> apply launch policy from manifest and CallConfig
  -> enqueue kernel(s) on runner stream
  -> enqueue copies or let explicit worker.copy_from handle output
  -> synchronize stream/event before returning to preserve current semantics

finalize:
  -> unload modules/libraries
  -> destroy streams/events
  -> free allocations
  -> release CUDA context ownership
  -> dlclose host runtime
```

### CUDA Persistent-Device Flow

```text
install/build time:
  -> build host runtime
  -> build persistent executor object archive

per callable:
  -> compile/link user task bodies into executor-compatible module
  -> generate dispatch table and metadata

prepare_callable:
  -> load linked executor module
  -> copy dispatch table and callable metadata to device

run_prepared:
  -> copy Runtime descriptor, TensorMap, ring, and args to global memory
  -> launch one persistent scheduler/worker kernel
  -> scheduler blocks/warps consume ready queues
  -> worker blocks/warps call device functions through dispatch table
  -> host synchronizes and copies outputs/diagnostics
```

The persistent runtime is the closer match for `tensormap_and_ringbuffer`, but
it requires a different callable contract. CUDA cannot load an arbitrary
global kernel from device code the way the Ascend scheduler hands work to
AICore executors. The CUDA executor needs linked device functions or a generated
dispatch layer inside the module launched by the host. See
[persistent-device.md](persistent-device.md) for the detailed analysis.

