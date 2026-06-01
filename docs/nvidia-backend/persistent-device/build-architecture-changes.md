# CUDA Persistent Device Runtime Analysis: Build and Architecture Changes

## Build and Architecture Changes

The current architecture assumes three target roles: `host`, `aicpu`,
`aicore`. CUDA needs runtime-specific target roles:

```python
BUILD_CONFIG = {
    "host": {...},
    "device_archive": {...},      # reusable persistent runtime objects
    "host_schedule": {...},       # optional host-schedule support module
}
```

Recommended build artifacts:

```text
build/lib/cuda/onboard/host_schedule/
  libhost_runtime.so
  libsimpler_log.so

build/lib/cuda/onboard/persistent_device/
  libhost_runtime.so
  libpto_cuda_persistent_device.a
  pto_persistent_manifest_schema.json
```

Per-callable build cache:

```text
build/cache/cuda/onboard/persistent_device/callables/{hash}/
  user_tasks.o
  generated_dispatch.o
  generated_manifest.o
  pto_callable.fatbin
  pto_callable.json
```

Required Python/build changes:

- continue replacing the global `TARGETS = ("host", "aicpu", "aicore")`
  assumption with per-runtime target discovery;
- continue generalizing runtime initialization around `RuntimeBinaries` roles;
  the current `role_paths` map exposes Ascend's legacy roles and CUDA's native
  `host` / optional `scheduler` / `device` roles, and CUDA no longer populates
  legacy AICPU/AICore path aliases;
- add a CUDA callable compiler that owns wrapper generation and final device
  link;
- add manifest fields to `ChipCallable` or introduce a CUDA callable payload
  format understood by the CUDA host runtime;
- keep `ChipWorker` C API stable unless the binary path generalization forces a
  wrapper-level compatibility shim.

