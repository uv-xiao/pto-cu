# NVIDIA Backend Flow Details: Sources

## Sources

- `simpler_setup/scene_test.py`
- `simpler_setup/kernel_compiler.py`
- `simpler_setup/runtime_builder.py`
- `python/simpler/worker.py`
- `python/simpler/task_interface.py`
- `src/common/worker/chip_worker.cpp`
- `src/common/worker/pto_runtime_c_api.h`
- NVIDIA CUDA Runtime API, Runtime vs Driver API:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html>
- NVIDIA CUDA Programming Guide, Driver API:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html>
- NVIDIA CUDA Runtime API, execution control:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html>
- NVIDIA CUDA Runtime API, library management:
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__LIBRARY.html>
- NVIDIA CUDA Driver API, module management:
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html>
- TileLang NVRTC wrapper docs:
  <https://tilelang.com/autoapi/tilelang/jit/adapter/nvrtc/wrapper/index.html>
- TileLang NVRTC library generator docs:
  <https://tilelang.com/autoapi/tilelang/jit/adapter/nvrtc/libgen/index.html>
- TileLang NVRTC compile helper docs:
  <https://tilelang.com/autoapi/tilelang/contrib/nvrtc/index.html>
