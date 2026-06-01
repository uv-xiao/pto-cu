# CUDA Build And Kernel Contract

## Directory Design

Add CUDA as a new architecture root rather than a variant under `a5`:

```text
src/cuda/
  docs/
    platform.md
    runtimes.md
  platform/
    include/
      common/
      host/
      cuda/
    onboard/
      host/
      device/
    sim/
      host/
      device/
    src/
      host/
  runtime/
    host_schedule/
      build_config.py
      host/
      runtime/
      orchestration/
    persistent_device/
      build_config.py
      host/
      device/
      runtime/
      orchestration/
```

The `onboard` variant means real CUDA hardware. The `sim` variant should only
be added if the project needs CPU-only CI for CUDA-specific code paths. It is
not required for phase 1 because CUDA compilation and module loading are the
main risk.

## Build System Changes

Platform discovery needs a third arch:

```python
PLATFORM_MAP = {
    "a2a3": ("a2a3", "onboard"),
    "a2a3sim": ("a2a3", "sim"),
    "a5": ("a5", "onboard"),
    "a5sim": ("a5", "sim"),
    "cuda": ("cuda", "onboard"),
}
```

The current `TARGETS = ("host", "aicpu", "aicore")` is Ascend-specific. CUDA
should introduce target roles that describe responsibilities rather than
hardware names:

- `host`: `libhost_runtime.so`, always present.
- `device`: CUDA cubin/fatbin/PTX for the executor or persistent runtime.
- `scheduler`: optional alias for the `persistent_device` scheduler module.

For a low-risk transition, `RuntimeBinaries` kept `aicpu_path` and
`aicore_path` temporarily as compatibility aliases. That transition has now
reached the role-map boundary with `RuntimeBinaries.role_paths` and
`RuntimeBinaries.path_for_role(...)`: Ascend exposes `host`, `aicpu`, and
`aicore`, while CUDA build configs now declare `host`, optional `scheduler`,
and `device`. CUDA `RuntimeBinaries` no longer fills `aicpu_path` or
`aicore_path`. The Python `ChipWorker` wrapper and `_ChipWorker` nanobind
boundary pass role-keyed maps directly when available. The C++ worker also
probes an optional `simpler_init_roles` host-runtime entry so CUDA runtimes can
receive `device` and `scheduler` binaries without pretending they are
AICPU/AICore images. The cleaner end state is a target-binary map keyed only
by runtime roles, with Ascend build configs declaring `host`, `aicpu`,
`aicore` and CUDA declaring `host`, optional `scheduler`, and `device`.

Add a `CudaNvccToolchain`:

- discovers `nvcc` from `CUDA_HOME`, `CUDA_PATH`, or `PATH`;
- accepts explicit SM targets, for example `sm_80`, `sm_90`, `sm_100`;
- can emit `--fatbin`, `--cubin`, or `--ptx`;
- forwards host compiler choice similarly to existing GCC handling.

Add an optional `CudaNvrtcToolchain` only when runtime compilation is needed.
Phase 1 can start with offline `nvcc` for deterministic builds.

`build_runtimes.py` should detect `cuda` when:

- `nvcc` is available for offline builds; and
- either a CUDA-capable device is visible or the requested phase only compiles
  fatbins without loading them.

## Kernel Compilation Contract

Today, `KernelCompiler.compile_incore()` assumes AIC/AIV source and returns a
binary uploaded to the device runtime. CUDA needs a separate path:

- `host_schedule`: compile one task body plus a generated `__global__`
  wrapper; the host runtime launches that wrapper by name.
- `persistent_device`: compile the same task body plus a generated device
  dispatch entry; the persistent executor calls it from worker warps/blocks.

Recommended public additions:

- `compile_cuda_host_schedule(source_path, entry_name, sm_targets)`
- `compile_cuda_persistent_device(sources, dispatch_manifest, sm_target)`

The existing `ChipCallable` can remain the language-level handle, but its
CUDA payload should include:

- module image bytes;
- entry names or lowered names;
- per-function argument layout;
- optional SM/compute target metadata;
- cache key based on source digest, compiler flags, and target SMs.
