# Gluon Tensor-Core GEMM

This note tracks the tensor-core `gluon-gen` milestone after scalar GEMM
correctness. It is correctness evidence only, not performance evidence.

## Current Contract

`KernelCompiler(platform="cuda").generate_gluon_kernel(...)` now supports
three Gluon tensor-core GEMM artifacts:

- `gemm_tensor_core_f16_f32`: one `64x32x32` FP16-input, FP32-output WGMMA
  correctness case.
- `gemm_tensor_core_tiled_f16_f32`: a tiled `64x128x32` CTA variant that loops
  over `K` in 32-wide blocks and launches a 2D grid over `M x N`.
- `gemm_tensor_core_tiled_bf16_f32`: the same tiled CTA shape with BF16
  inputs and FP32 accumulator/output. Its review harness covers a smoke tile
  and a bounded linear-style `m=64,k=7168,n=128` shape using
  `DeepSeek-V4-Flash config hidden_size=7168` provenance.

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

The BF16 serving-shape fixture is generated and skip-safe, but the current
remote H200 Gluon stack did not expose the Hopper WGMMA APIs needed to run it.
The command below was run through `--sync` against the remote checkout with a
project-local `.venv`.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_gemm_tensor_core_bf16.py \
      --output-dir tmp/gluon-tensor-core-bf16-h200 \
      --arch compute_90 --sweep --require-cuda'
```

Distilled result:

```json
{
  "kernel_name": "gemm_tensor_core_tiled_bf16_f32",
  "status": "skipped",
  "case_count": 2,
  "passed_cases": 0,
  "skipped_cases": 2,
  "cases": [
    {
      "case_name": "smoke_tile",
      "shape": {"m": 64, "n": 128, "k": 32},
      "dtype_boundary": {
        "a": "bfloat16",
        "b": "bfloat16",
        "accumulator": "float32",
        "out": "float32"
      },
      "reason": "missing warpgroup_mma and NVMMADistributedLayout"
    },
    {
      "case_name": "deepseek_v4_flash_hidden_size",
      "shape": {"m": 64, "n": 128, "k": 7168},
      "provenance": "DeepSeek-V4-Flash config hidden_size=7168",
      "dtype_boundary": {
        "a": "bfloat16",
        "b": "bfloat16",
        "accumulator": "float32",
        "out": "float32"
      },
      "reason": "missing warpgroup_mma and NVMMADistributedLayout"
    }
  ]
}
```

The direct H200 probe reported `NVIDIA H200 NVL`, compute capability `9.0`,
driver `580.126.20`, Triton `3.4.0`, `gl.bfloat16` present, but
`gl.NVMMADistributedLayout` absent and
`triton.experimental.gluon.language.nvidia.hopper.warpgroup_mma` unavailable.
This is unsupported-API evidence, not BF16 runtime correctness evidence.

## Boundary

This PR does not include flash-attention evidence, MoE or distributed
evidence, serving evidence, vLLM/simpler-nv integration, or a performance
claim. The BF16 fixture is not a FlashInfer integration claim and not a
production-readiness claim. Generated tensor-core source and JSON manifests
stay under `tmp/gluon-*`.
