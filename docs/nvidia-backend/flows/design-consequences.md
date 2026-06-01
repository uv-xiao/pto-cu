# NVIDIA Backend Flow Details: Design Consequences

## Design Consequences

- CUDA support should start with `host_schedule`; it matches CUDA's natural
  host-launched model and proves the stable C API.
- `tensormap_and_ringbuffer` should not be ported by mechanically renaming
  AICPU/AICore roles. It needs a persistent-kernel design and a linked device
  dispatch table.
- `RuntimeBinaries` is now target-role based. Ascend still exposes its legacy
  `aicpu`/`aicore` roles, while CUDA exposes `host`, `device`, and optional
  `scheduler` without populating AICPU/AICore aliases. The Python and nanobind
  `ChipWorker` boundary accepts role-keyed maps directly, and the CUDA
  `persistent_device` build publishes a `scheduler` role so scheduler/runtime
  artifacts do not have to pass through AICPU naming. The C++ `ChipWorker` now
  probes the optional `simpler_init_roles` host-runtime ABI and passes CUDA
  `device`/`scheduler` role binaries directly when that entry exists; runtimes
  without it keep the legacy two-binary `simpler_init` path.
- The backend should use Driver API for module/context ownership because PTO
  is a dynamic plugin-like runtime loaded through `dlopen`.
- TileLang validates the NVRTC + Driver API model for dynamic CUDA code, but
  PTO should keep C++ host-runtime ownership rather than generating Python
  launch wrappers.

