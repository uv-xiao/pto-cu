# CUDA Backend Status: Platform And Runtime Discovery

## Platform And Runtime Discovery

- `cuda` maps to the `cuda/onboard` platform variant.
- `RuntimeBuilder(platform="cuda")` discovers both CUDA runtimes:
  `host_schedule` and `persistent_device`.
- `RuntimeBinaries` now exposes a role-keyed view through `role_paths` and
  `path_for_role(...)`. Ascend platforms map the existing `host`, `aicpu`,
  and `aicore` roles directly. CUDA build configs now declare native `host`
  and `device` targets, and `persistent_device` also declares a native
  `scheduler` target. CUDA `RuntimeBinaries` no longer populates legacy
  `aicpu_path` / `aicore_path` aliases; consumers use the role map instead.
- The Python `ChipWorker` wrapper and underlying `_ChipWorker` nanobind
  boundary now accept role-keyed runtime binary maps directly through
  `_ChipWorker.init_roles(...)`. The C++ `ChipWorker` probes the optional
  host-runtime `simpler_init_roles(...)` ABI and passes non-host CUDA roles
  such as `device` and `scheduler` without lowering them into AICPU/AICore
  slots. The legacy positional init remains as a compatibility fallback for
  runtimes that do not export the optional role-keyed entry.

Evidence:

- `tests/ut/py/test_cuda_backend.py`
- `tests/ut/py/test_runtime_builder.py`
- `tests/ut/py/test_runtime_compiler.py`
- `src/cuda/runtime/host_schedule/build_config.py`
- `src/cuda/runtime/persistent_device/build_config.py`
- `src/cuda/platform/onboard/device/CMakeLists.txt`
- `src/cuda/platform/onboard/scheduler/CMakeLists.txt`
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp`

