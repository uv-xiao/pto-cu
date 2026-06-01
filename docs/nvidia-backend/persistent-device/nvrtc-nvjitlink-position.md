# CUDA Persistent Device Runtime Analysis: NVRTC and nvJitLink Position

## NVRTC and nvJitLink Position

NVRTC plus nvJitLink can support runtime specialization, but they should not be
the first persistent-device foundation. The offline `nvcc` path is easier to
make reproducible because it uses the same compiler/link flow developers run
locally and in CI.

Use NVRTC/nvJitLink later only if there is a clear need:

- shape-specialized kernels generated at runtime;
- autotuned schedules selected after install;
- avoiding full `nvcc` invocation latency for small generated tasks.

Until then, use:

- `nvcc -dc` / `-rdc=true` for separable device objects;
- normal device link to produce cubin/fatbin;
- optional device LTO once correctness and link structure are stable.

