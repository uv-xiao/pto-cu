# Gluon Tensor-Core GEMM

This note tracks the tensor-core `gluon-gen` milestone after scalar GEMM
correctness. It is correctness and unsupported-boundary evidence only, not
performance evidence.

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
- `gemm_tensor_core_tiled_fp8e4nv_f32`: the same tiled CTA shape with
  `torch.float8_e4m3fn` inputs mapped to Gluon `gl.float8e4nv` operands and
  FP32 accumulator/output. This is a boundary harness only because H200
  lowering currently rejects the generated FP8 WGMMA type/shape.
- `examples/cuda/gluon_gemm_tensor_core_fp4.py`: an FP4 dtype API boundary
  harness for the next low-precision GEMM gate. It records Torch and Gluon
  FP4 attrs and does not generate source when no Gluon FP4 WGMMA operand dtype
  is available.

The source-generating FP16 harnesses emit source and manifest artifacts before
checking runtime CUDA availability. BF16 and FP8 source-generating harnesses
emit source and manifest artifacts before their runtime or lowering checks.
They emit structured JSON in pass, skip, and generation-failure paths.
`--require-cuda` returns non-zero when CUDA, PyTorch, Triton/Gluon Hopper WGMMA
APIs, generation, or correctness are unavailable.

FP4 remains a dtype/API boundary harness: it records the discovered Torch and
Gluon FP4 attrs and does not generate source or manifest artifacts unless a
confirmed Gluon FP4 WGMMA operand dtype path exists. Its `--require-cuda`
failure is review-visible unsupported-boundary evidence, not a generation or
correctness artifact.

`examples/cuda/gluon_wgmma_api_preflight.py` is the repo-owned API gate for
future BF16, FP8, FP4, and grouped GEMM WGMMA work. It emits structured JSON
covering Torch import and CUDA/Hopper visibility, Triton version, Gluon import
status, `TensorDescriptor`, the Hopper WGMMA primitive imports, and the Gluon
layout/dtype attributes used by generated tensor-core source. With
`--require-cuda`, missing CUDA/Hopper or missing WGMMA APIs return non-zero
before a kernel harness attempts runtime correctness.

The preflight payload also reports discovered Torch and Gluon FP8 and FP4
dtype/API attributes. The FP8 and FP4 harnesses record those dtype probes in
JSON before attempting WGMMA lowering, so missing or ambiguous low-precision
dtype support is a review-visible boundary instead of a fake correctness pass.

The generated source uses Hopper Gluon APIs:

- `TensorDescriptor` for tensor descriptors.
- `NVMMASharedLayout` for shared-memory operand tiles.
- `NVMMADistributedLayout` for the accumulator layout.
- `warpgroup_mma` and `warpgroup_mma_wait` for WGMMA tensor-core execution.

## H200 Correctness Evidence

The commands below passed on a remote H200 host through the generic tree-sync
CUDA runner. The FP16 checks used a project-local `.venv` with Torch, CUDA,
Triton, and Gluon Hopper WGMMA APIs available.

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

The BF16 correctness check first ran the WGMMA preflight gate in a fresh
project-local `.venv`. The fresh project-local `.venv` preflight failed
because Torch and Triton were not installed in that environment, so the result
is environment-missing evidence only:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'python3 -m venv --system-site-packages .venv && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/gluon_wgmma_api_preflight.py --require-cuda'
```

Distilled fresh-venv result:

```json
{
  "status": "failed",
  "cuda_required_missing": ["torch", "cuda_available", "hopper_device"],
  "missing_required": [
    "TensorDescriptor",
    "warpgroup_mma",
    "warpgroup_mma_wait",
    "mbarrier",
    "tma",
    "fence_async_shared",
    "gl.NVMMASharedLayout",
    "gl.NVMMADistributedLayout",
    "gl.bfloat16",
    "triton",
    "gluon"
  ]
}
```

The same synced checkout then used the preserved Gluon environment from the
previous WGMMA API-gate work:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  <remote-gluon-venv>/bin/python \
    examples/cuda/gluon_wgmma_api_preflight.py --require-cuda
```

Distilled preserved-env preflight result:

```json
{
  "status": "passed",
  "reason": "all required WGMMA APIs are available",
  "cuda": {
    "device_count": 8,
    "selected_device": {
      "index": 0,
      "name": "NVIDIA H200 NVL",
      "capability": [9, 0]
    }
  },
  "triton": {"imported": true, "version": "3.7.1"},
  "missing_required": [],
  "cuda_required_missing": []
}
```

The preserved Gluon environment preflight passed with Triton `3.7.1`. Its
Python environment is recorded as
`python environment: <remote-gluon-venv>/bin/python`; concrete remote paths are
intentionally omitted. After that API gate passed, the BF16 sweep command was:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  <remote-gluon-venv>/bin/python \
    examples/cuda/gluon_gemm_tensor_core_bf16.py \
    --output-dir tmp/gluon-tensor-core-bf16-h200 \
    --arch compute_90 --sweep --require-cuda
```

Distilled BF16 sweep result:

```json
{
  "kernel_name": "gemm_tensor_core_tiled_bf16_f32",
  "status": "passed",
  "case_count": 2,
  "passed_cases": 2,
  "failed_cases": 0,
  "skipped_cases": 0,
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
      "status": "passed",
      "max_abs_error": 3.814697265625e-06
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
      "status": "passed",
      "max_abs_error": 0.002899169921875
    }
  ]
}
```

The BF16 case statuses were `passed, passed`. The largest absolute error was
`0.002899169921875` on the `m=64,n=128,k=7168` linear-style case. The direct
H200 probe for this run reported `NVIDIA H200 NVL`, compute capability `9.0`,
driver `580.126.20`, and 143771 MiB memory per GPU.

## H200 FP8 Boundary Evidence

The next FP8 gate used the same synced checkout and preserved Gluon
environment. The general WGMMA preflight still passed on H200 with Triton
`3.7.1`. A targeted dtype probe found Gluon FP8 dtype attrs
`float8e4b15`, `float8e4b8`, `float8e4nv`, `float8e5`, and `float8e5b16`,
all with `primitive_bitwidth=8`. The same environment exposed Torch FP8
dtypes including `float8_e4m3fn`.

The committed FP8 boundary command was:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  <remote-gluon-venv>/bin/python \
    examples/cuda/gluon_gemm_tensor_core_fp8.py \
    --output-dir tmp/gluon-tensor-core-fp8-h200 \
    --arch compute_90 --require-cuda
```

The command returned exit status `1`, which is expected for the unsupported
boundary. Distilled JSON stdout:

```json
{
  "kernel_name": "gemm_tensor_core_tiled_fp8e4nv_f32",
  "status": "failed",
  "shape": {"m": 64, "n": 128, "k": 32},
  "dtype_boundary": {
    "a": "torch.float8_e4m3fn / gl.float8e4nv",
    "b": "torch.float8_e4m3fn / gl.float8e4nv",
    "accumulator": "float32",
    "out": "float32"
  },
  "fp8_dtype_probe": {
    "status": "passed",
    "gl_dtype": "float8e4nv",
    "torch_dtype": "float8_e4m3fn",
    "gl_primitive_bitwidth": 8
  },
  "unsupported_boundary": {
    "kind": "gluon_fp8_wgmma_compile",
    "expected_failure": "Triton/Gluon may reject FP8 WGMMA type or shape lowering"
  },
  "error_type": "RuntimeError",
  "error": "PassManager::run failed"
}
```

The compiler stderr explains the boundary more specifically: Triton/Gluon
lowered the operands as `f8E4M3FN` with MMA instruction shape `[16, 16, 32]`,
then failed in `ConvertNVGPUToLLVM` with the assertion
`WGMMA type or shape is not supported`. This is FP8 API and unsupported
lowering evidence only. It is not FP8 GEMM correctness evidence.

## H200 FP4 Boundary Evidence

The next FP4 gate used the same synced checkout and preserved Gluon
environment. The general WGMMA preflight still passed on H200 with Triton
`3.7.1`. It found Torch FP4 dtype `torch.float4_e2m1fn_x2`. On the Gluon side,
the only FP4-related attr found was the helper `fp4_to_fp`; candidate dtype
attrs `float4_e2m1fn`, `float4_e2m1fn_x2`, and `float4e2m1` were absent, so
there is no confirmed Gluon FP4 WGMMA operand dtype.

The committed FP4 boundary command was:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  <remote-gluon-venv>/bin/python \
    examples/cuda/gluon_gemm_tensor_core_fp4.py \
    --output-dir tmp/gluon-tensor-core-fp4-h200 \
    --arch compute_90 --require-cuda
```

The command returned exit status `2`, which is expected for a required-CUDA
skip at this unsupported dtype API boundary. Distilled JSON stdout:

```json
{
  "kernel_name": "gemm_tensor_core_tiled_fp4_f32",
  "status": "skipped",
  "artifact": null,
  "shape": {"m": 64, "n": 128, "k": 32},
  "dtype_boundary": {
    "a": "torch.float4_e2m1fn_x2 / Gluon FP4 WGMMA dtype unavailable",
    "b": "torch.float4_e2m1fn_x2 / Gluon FP4 WGMMA dtype unavailable",
    "accumulator": "float32",
    "out": "float32"
  },
  "fp4_dtype_probe": {
    "status": "failed",
    "torch_dtype": "float4_e2m1fn_x2",
    "torch_fp4_attrs": ["float4_e2m1fn_x2"],
    "gl_dtype": null,
    "gl_fp4_attrs": ["fp4_to_fp"],
    "reason": "missing Gluon FP4 WGMMA dtype API"
  },
  "unsupported_boundary": {
    "kind": "gluon_fp4_dtype_api_unavailable",
    "expected_failure": "Torch exposes packed FP4, but no Gluon FP4 WGMMA dtype is available"
  }
}
```

This is FP4 API and unsupported-boundary evidence only. It is not FP4 GEMM
correctness evidence.

## Boundary

This PR does not include flash-attention evidence, MoE or distributed
evidence, serving evidence, vLLM/simpler-nv integration, or a performance
claim. The BF16 fixture is not FlashInfer integration evidence, not
generated-kernel performance evidence, not vLLM/simpler-nv serving integration
evidence, and not production-readiness evidence. The FP8 boundary is not
FlashInfer integration evidence, not serving integration evidence, not
generated-kernel performance evidence, not production-readiness evidence, and
not BF16/FP4/grouped GEMM/MoE/FlashAttention/vLLM integration evidence. The
FP4 boundary is not FlashInfer integration evidence, not serving integration
evidence, not generated-kernel performance evidence, not production-readiness
evidence, and not BF16/FP8/grouped GEMM/MoE/FlashAttention/vLLM integration
evidence. Generated tensor-core source and JSON manifests stay under
`tmp/gluon-*` when source generation is supported by the dtype/API boundary.
