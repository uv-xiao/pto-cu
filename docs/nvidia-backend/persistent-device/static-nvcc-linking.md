# CUDA Persistent Device Runtime Analysis: Static NVCC Linking Feasibility

## Static NVCC Linking Feasibility

The stable path is feasible with ordinary `nvcc`, but it changes what "runtime
binary" means for `persistent_device`.

CUDA supports separate compilation of device code. Device functions can call
device functions and access device variables from other compilation units when
all CUDA translation units are compiled with relocatable device code enabled
(`-dc` or `-rdc=true`) and then device-linked. This gives PTO a stable offline
path:

```text
build reusable runtime object/archive:
  pto_persistent_executor.cu
  pto_scheduler.cu
  pto_runtime_state.cu
    -> nvcc -dc ... -> libpto_cuda_persistent_device.a

per callable:
  user_tasks.cu
  user_orchestrator_device.cu
  generated_dispatch.cu       # rendered from task func_id/name/body metadata
  generated_manifest.cu
    -> nvcc -dc ...
    -> nvcc device link with libpto_cuda_persistent_device.a
    -> cubin or fatbin

prepare_callable:
  -> host runtime loads final cubin/fatbin
  -> copies manifest/runtime metadata to device

run_prepared:
  -> launches pto_persistent_executor from that final module
```

The cost is that `persistent_device` needs a per-callable device link step.
Unlike Ascend, the fully linked scheduler/worker executor cannot be entirely
prebuilt independently of user task code if worker code calls user functions
directly.

That is acceptable if the build system makes the split explicit:

- `RuntimeBuilder` builds reusable host runtime and reusable CUDA device object
  archives.
- `KernelCompiler` or a CUDA-specific callable compiler compiles user tasks,
  generated dispatch, and optional device orchestration.
- The callable compiler links those objects with the runtime archive into the
  final module image stored in the callable payload.

