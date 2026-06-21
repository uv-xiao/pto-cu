# Gluon Tensor-Core GEMM

This note tracks the tensor-core `gluon-gen` milestone after scalar GEMM
correctness. It is correctness evidence only, not performance evidence.

## Current Contract

`KernelCompiler(platform="cuda").generate_gluon_kernel(...)` now supports two
Gluon tensor-core GEMM artifacts:

- `gemm_tensor_core_f16_f32`: one `64x32x32` FP16-input, FP32-output WGMMA
  correctness case.
- `gemm_tensor_core_tiled_f16_f32`: a tiled `64x128x32` CTA variant that loops
  over `K` in 32-wide blocks and launches a 2D grid over `M x N`.

Both harnesses always generate source and manifest artifacts before checking
runtime CUDA availability. They emit structured JSON in pass, skip, and
generation-failure paths. `--require-cuda` returns non-zero when CUDA, PyTorch,
Triton/Gluon Hopper WGMMA APIs, generation, or correctness are unavailable.

The generated source uses Hopper Gluon APIs:

- `TensorDescriptor` for tensor descriptors.
- `NVMMASharedLayout` for shared-memory operand tiles.
- `NVMMADistributedLayout` for the accumulator layout.
- `warpgroup_mma` and `warpgroup_mma_wait` for WGMMA tensor-core execution.

## H200 Correctness Evidence

The commands below passed on a remote H200 host through the generic tree-sync
CUDA runner. The remote temporary checkout used a project-local `.venv` with
Torch, CUDA, Triton, and Gluon Hopper WGMMA APIs available.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_gemm_tensor_core.py \
      --output-dir tmp/gluon-tensor-core-h200 \
      --arch compute_90 --require-cuda'
```

Distilled result:

```json
{
  "kernel_name": "gemm_tensor_core_f16_f32",
  "status": "passed",
  "shape": {"m": 64, "n": 32, "k": 32},
  "tolerance": {"atol": 0.001, "rtol": 0.1},
  "max_abs_error": 0.0073032379150390625,
  "tile_shape": [64, 32, 32]
}
```

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_gemm_tensor_core_tiled.py \
      --output-dir tmp/gluon-tensor-core-tiled-h200 \
      --m 256 --n 256 --k 64 \
      --arch compute_90 --require-cuda'
```

Distilled result:

```json
{
  "kernel_name": "gemm_tensor_core_tiled_f16_f32",
  "status": "passed",
  "shape": {"m": 256, "n": 256, "k": 64},
  "tolerance": {"atol": 0.001, "rtol": 0.1},
  "max_abs_error": 0.009243011474609375,
  "tile_shape": [64, 128, 32]
}
```

## Boundary

This PR does not include flash-attention evidence, MoE or distributed
evidence, serving evidence, or a performance claim. Generated tensor-core
source and JSON manifests stay under `tmp/gluon-*`.
