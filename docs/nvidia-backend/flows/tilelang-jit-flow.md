# NVIDIA Backend Flow Details: TileLang JIT Flow

## TileLang JIT Flow

TileLang is useful prior art because it has both ordinary TVM-backed JIT and
an NVRTC backend for fast CUDA JIT execution.

Observed TileLang NVRTC flow:

```text
TileLang Python/TIR function
  |
  | TVM/TileLang lowering
  v
CUDA source string
  |
  | tilelang.contrib.nvrtc.compile_cuda(...)
  v
PTX or cubin bytes
  |
  | NVRTCLibraryGenerator.compile_lib()
  v
temp files:
  kernel.cu
  kernel.cubin
  launcher.py
  |
  | NVRTCLibraryGenerator.load_lib()
  v
CUDA Driver API library/module handle + generated Python launcher
  |
  | generated call(...)
  v
cuda.bindings.driver:
  cuKernelSetAttribute
  cuLaunchKernelEx
  optional TMA descriptors and L2 cache setup
```

TileLang's NVRTC path compiles CUDA source at runtime to PTX or cubin, writes
a generated Python launcher, loads the cubin with CUDA Driver API bindings,
and launches through `cuLaunchKernelEx`. The wrapper also generates setup code
for TMA descriptors, L2 persistence, and launch attributes.

What PTO should copy from TileLang:

- specialize and compile user code close to runtime when launch metadata
  depends on shapes or tuning decisions;
- keep a manifest/launcher layer separate from raw code bytes;
- use Driver API module/kernel handles for dynamically loaded CUDA code;
- cache compiled artifacts and function handles by target architecture and
  source/config digest.

What PTO should not copy directly:

- TileLang generates Python launchers; PTO should keep launch execution inside
  `libhost_runtime.so` because `ChipWorker` already centralizes device
  lifecycle in C++.
- TileLang is operator-kernel oriented; PTO also needs DAG/task runtime
  semantics, callable IDs, `TaskArgs`, and a stable C API boundary.

