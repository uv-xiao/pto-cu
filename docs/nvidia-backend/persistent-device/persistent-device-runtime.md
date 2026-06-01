# CUDA Persistent Device Runtime Analysis: Persistent Device Runtime

## Persistent Device Runtime

The `persistent_device` runtime is the CUDA analogue of
`tensormap_and_ringbuffer`, not a port of the AICPU implementation.

The first implementation slice in this branch is intentionally a tracer
bullet: `persistent_device` is registered as a CUDA runtime and can launch a
single executor kernel that consumes a device array of vector-add task
descriptors. It proves the build/discovery path, module loading, descriptor
memory layout, and "one host launch handles multiple device tasks" shape before
adding TensorMap/ring queues and generated dispatch tables.

The next implementation slice adds a bounded ready ring for the same vector-add
task descriptor. One scheduler block publishes task IDs into global memory and
worker blocks pop those IDs through atomics inside the same executor launch.
The queue uses per-slot sequence flags so a capacity smaller than the task
count can wrap without a later ticket consuming an earlier slot value. This
still is not the final TensorMap/ring runtime, but it exercises the
scheduler/worker split and back-pressure shape that CUDA needs because there
is no AICPU.

The following slice layers a small DAG on top of that bounded ring. Task
descriptors carry a `func_id`, dependent ranges, an initial fan-in count, and
optional tensor shape/stride metadata for tiled callables. The persistent
executor seeds zero-fan-in tasks, dispatches task bodies through a generated
`func_id` switch, decrements dependent fan-in counters when tasks complete,
pushes newly ready dependents back into the ring, and keeps one scheduler
block alive to report malformed graphs that exhaust ready work before every
task completes. The smoke path now uses
`KernelCompiler(platform="cuda").compile_cuda_persistent_device(...)` to
generate the shared task-body wrappers and dispatch switch before compiling
the executor source with `nvcc`. The compiler writes the generated source, PTX,
and JSON manifest under
`build/cache/cuda/onboard/persistent_device/callables/`, matching the intended
per-callable artifact layout.

### Runtime Roles

```text
Persistent executor grid

block 0..S-1:
  scheduler role
  - drain submit/wiring queues
  - compute ready tasks
  - push task descriptors into device ready queues

block S..N-1:
  worker role
  - pop descriptors
  - call generated dispatch(func_id, args)
  - publish completion records
```

The roles can also be assigned by warp instead of block. Warp roles are more
flexible but harder to reason about because scheduler and worker warps share a
CTA's synchronization and shared memory. Block roles are the safer first
implementation.

### Dispatch Without Child Kernels

The executor should not launch `__global__` task kernels from device code.
Instead it calls task functions compiled into the same device image:

```cuda
using PtoTaskFn = void (*)(PtoTaskContext *);

__device__ void vector_add_task(PtoTaskContext *ctx) {
    // user task body
}

__device__ void pto_dispatch(uint32_t func_id, PtoTaskContext *ctx) {
    switch (func_id) {
    case 0:
        vector_add_task(ctx);
        return;
    default:
        pto_fail_unknown_func(func_id);
    }
}

__global__ void pto_persistent_executor(PtoRuntimeState *state) {
    if (is_scheduler_block()) {
        pto_scheduler_loop(state);
    } else {
        pto_worker_loop(state, pto_dispatch);
    }
}
```

A generated `switch` is the recommended first dispatch mechanism. A table of
`__device__` function pointers is possible on CUDA, but it is harder to debug,
less friendly to optimization, and more sensitive to relocation/linking
details. A generated switch also lets `nvcc` inline simple task bodies when
whole-program or LTO compilation can see them.

