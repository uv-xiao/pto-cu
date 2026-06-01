# NVIDIA Backend Flow Details: Lifecycle, Memory, and Callable Mapping

## Lifecycle, Memory, and Callable Mapping

### Lifecycle

| PTO C API concept | a2a3/a5 implementation | CUDA implementation |
| ----------------- | ---------------------- | ------------------- |
| `create_device_context` | heap-allocate `DeviceRunner` | heap-allocate CUDA `DeviceRunner` |
| `simpler_init` | `rtSetDevice`, set log state | retain/set CUDA context, create streams |
| `finalize_device` | destroy streams, free device resources | unload modules, destroy streams, free memory |
| `destroy_device_context` | delete runner | delete runner and release context ref |
| host runtime `dlopen` | `libhost_runtime.so` per runtime | same ABI, CUDA-specific implementation |

CUDA detail: `simpler_init` should make the CUDA context current for the
calling thread or retain the primary context and make it current before each
operation. This mirrors the existing rule that device operations happen on the
thread that initialized the `ChipWorker`, while still allowing future
thread-local context attachment if the Python worker model needs it.

### Memory

| PTO operation | a2a3/a5 | CUDA |
| ------------- | ------- | ---- |
| `device_malloc_ctx` | `DeviceRunner::allocate_tensor` -> Ascend allocator | `cuMemAlloc` or `cudaMalloc` |
| `device_free_ctx` | Ascend allocator free | `cuMemFree` or `cudaFree` |
| `copy_to_device_ctx` | H2D copy through CANN runtime | `cuMemcpyHtoDAsync` or `cudaMemcpyAsync` |
| `copy_from_device_ctx` | D2H copy through CANN runtime | `cuMemcpyDtoHAsync` or `cudaMemcpyAsync` |
| `ContinuousTensor.data` | device pointer integer | `CUdeviceptr` / device pointer integer |
| `child_memory` | skip host copy for child-owned device ptr | same semantic: pointer already device-owned |

PTO should preserve explicit copy semantics at the `ChipWorker` layer. Unified
memory may be useful later, but it should not be the first backend contract
because current tests/examples reason about explicit allocation and copy.

### Callable

| PTO callable concept | a2a3/a5 | CUDA `host_schedule` | CUDA `persistent_device` |
| -------------------- | ------- | -------------------- | ------------------------ |
| `ChipCallable.binary` | orchestration shared object bytes | module image or host-scheduler metadata | linked executor module image |
| `CoreCallable.binary` | AICore text/object or sim SO | CUDA cubin/PTX/fatbin bytes | task body object or IR |
| `func_id` | runtime dispatch id for AICore kernels | entry-name or kernel-handle id | dispatch-table index |
| `prepare_callable` | upload kernels + orch SO; cache by cid | load module; cache kernel handles | load linked executor and table |
| `run_prepared` | launch scheduler/executor pair | launch one or more CUDA kernels | launch persistent executor |
| `unregister_callable` | drop prepared state/refcounts | unload module when no cid references it | unload module and free table state |

The CUDA callable payload should include enough metadata to avoid guessing at
launch time:

- target architecture and image kind (`ptx`, `cubin`, `fatbin`);
- entry names and stable IDs;
- argument ABI for each entry;
- launch policy defaults;
- optional dynamic shared-memory and cluster attributes;
- source/config digest for cache lookup.

### `CallConfig`

Current `CallConfig` is Ascend-shaped but can remain the wire format during
bring-up:

- `block_dim`: use as a launch-policy hint in `host_schedule`, and as the
  persistent executor block count in the future runtime.
- `aicpu_thread_num`: treat as ignored or as a scheduler-role hint for CUDA;
  document exact behavior per runtime.
- diagnostic flags: map to CUDA-side profiler/tensor dump support only when
  those features exist; otherwise fail with a clear not-supported error rather
  than silently producing incomplete artifacts.

Longer term, add neutral aliases while keeping the binary layout compatible:
for example `worker_blocks` for `block_dim` and `scheduler_lanes` for
`aicpu_thread_num`.

