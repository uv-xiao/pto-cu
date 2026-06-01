# CUDA Semantics, Testing, And Roadmap

## Runtime Semantics Mapping

| Existing concept | CUDA `host_schedule` mapping | CUDA `persistent_device` mapping |
| ---------------- | ---------------------------- | -------------------------------- |
| AICPU scheduler | Host code in `libhost_runtime.so` | Scheduler warps/blocks in executor |
| AICore worker | CUDA global kernel wrapper | Worker warps/blocks in executor |
| `block_dim` | grid/block policy hint | persistent block count |
| `aicpu_thread_num` | ignored or scheduler stream count | scheduler-role count |
| kernel upload | `cuModuleLoadDataEx` / runtime library load | same, plus dispatch table |
| task completion | stream/event synchronization | global-memory completion queue |
| device graph build | not in phase 1 | CUDA global-memory TensorMap/ring |

`CallConfig` should gain CUDA-neutral names before the backend matures. The
current `aicpu_thread_num` name can be accepted for compatibility, but CUDA
documentation should call it a scheduler-worker hint and define exact behavior
per runtime.

## Testing Strategy

Phase 1 tests:

- Python unit tests for platform discovery and `cuda` build-target selection.
- C++ unit tests for CUDA host runtime stubs where CUDA headers are available.
- A smoke scene test that allocates two tensors, launches vector add, copies
  results back, and validates output on one CUDA device.
- A no-device test path that verifies `cuda` is skipped with a clear message
  when `nvcc` or the driver is missing.

Phase 2 tests:

- persistent executor unit test with a synthetic ready queue;
- dispatch-table test that invokes two different user functions by id;
- stress test for fanin/fanout counters and queue wraparound;
- comparison against `host_schedule` for the same DAG.

CI should initially treat CUDA as optional and report skipped tests clearly.

## Open Decisions

- Whether phase 1 should use CUDA Runtime API, Driver API, or a thin mix. The
  Driver API is better aligned with dynamic module loading; Runtime API is
  simpler for memory and streams.
- Whether CUDA user kernels should be `.cu` files beside existing examples or
  generated from a portable PTO kernel DSL.
- Whether `RuntimeBinaries` should be generalized before CUDA lands, or
  whether CUDA should temporarily fit into the existing three-path ABI.
- Which SM targets are required for the first deployment environment.
- Whether CUDA Graphs should be used for repeated `host_schedule` launches once
  correctness is established.

## Recommended First Implementation Slice

1. Add platform discovery for `cuda` behind toolchain detection.
2. Add `CudaNvccToolchain` and host/device target compilation.
3. Add `src/cuda/platform/onboard/host` exporting the existing C API.
4. Implement memory allocation, copies, init/finalize, and no-op comm stubs.
5. Port `host_schedule` only.
6. Add one vector-add scene test and skip it when CUDA is unavailable.
7. Revisit `RuntimeBinaries` and device-object packaging before starting
   `persistent_device`.

This keeps the first branch focused on proving the ABI and build integration.
It postpones the hardest CUDA-specific scheduling work until the project has a
real CUDA `ChipWorker` path running end to end.
