# CUDA Backend Architecture And Scope

## Current Runtime Shape

The current chip backend has three separately built programs:

- `libhost_runtime.so`: loaded by `ChipWorker` through `dlopen`; owns device
  context, device memory, copies, profiling hooks, and launch orchestration.
- `libaicpu_kernel.so`: device-side scheduler running on Ascend AICPU.
- `aicore_kernel.o` or sim `.so`: device worker executor that receives task
  dispatches and invokes user kernels.

The build and lookup path is:

1. `simpler_setup.platform_info` maps a platform name to `(arch, variant)`.
2. `RuntimeBuilder` discovers `src/{arch}/runtime/*/build_config.py`.
3. `RuntimeCompiler` builds `host`, `aicpu`, and `aicore` targets.
4. `ChipWorker` loads the host runtime and passes the device binaries through
   the stable C API in `src/common/worker/pto_runtime_c_api.h`.

Runtime implementations are independent of platform implementations. Platform
code lives under `src/{arch}/platform/`, while runtime graph/scheduler code
lives under `src/{arch}/runtime/`. The two current runtime variants are:

- `host_build_graph`: host builds the task graph; useful for bring-up.
- `tensormap_and_ringbuffer`: production runtime where AICPU-side
  orchestration derives dependencies and dispatches tasks to AICore workers.

L3+ hierarchical scheduling is separate from the L2 device backend. It sees
`ChipWorker` as an `IWorker` leaf and passes `Callable`, `TaskArgsView`, and
`CallConfig` through the same interface regardless of device vendor.

## CUDA Constraints That Matter

CUDA has a different control model from Ascend:

- CUDA systems are heterogeneous: host CPU code uses CUDA APIs to allocate
  device memory, copy data, launch GPU kernels, and synchronize.
- A GPU consists of Streaming Multiprocessors (SMs). Kernels launch many
  threads organized into thread blocks and grids. Blocks are scheduled onto
  SMs, and independent blocks cannot rely on a global execution order.
- Threads execute in warps of 32 lanes under SIMT. Blocks share fast on-chip
  shared memory; global memory is visible across the device.
- CUDA launches and copies are asynchronous. Streams express ordered queues of
  operations, and synchronization is explicit.
- Offline compilation uses `nvcc`, which separates host and device code,
  emits PTX, assembles target-specific cubins with `ptxas`, and can package
  multiple PTX/cubin images into a fatbin.
- The Driver API can load PTX, cubin, or fatbin modules and resolve kernel
  handles. PTX is the forward-compatible form; cubin is tied to an SM target.
- NVRTC and nvJitLink enable runtime compilation and linking. They are useful
  for user kernels, but they add toolkit/runtime compatibility constraints.

The biggest architectural mismatch is the missing AICPU equivalent. NVIDIA
does not expose a separate device control CPU that can `dlopen` orchestration
code, write hardware dispatch registers for another core type, and invoke
arbitrary separately compiled global kernels. The CUDA backend should treat
that as a hard boundary and avoid copying the AICPU/AICore split literally.

## Naming and Scope

Use `cuda` as the backend name in code and platform strings:

- `src/cuda/platform/`
- `src/cuda/runtime/`
- platform: `cuda`
- optional future host-only simulator: `cudasim`

Use "NVIDIA backend" in prose when referring to hardware/vendor support, and
`cuda` for concrete paths, build targets, and CLI platform names. The Python
package name remains `simpler`.

Phase 1 should target a single NVIDIA GPU on one host. Multi-GPU and
multi-node communication should be a later phase using NCCL or CUDA IPC behind
the existing `comm_*` C API surface.
