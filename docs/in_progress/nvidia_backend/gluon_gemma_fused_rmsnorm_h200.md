# Gluon Gemma Fused RMSNorm FP32 H200 Correctness

This note records correctness evidence for the generated Gluon
`gemma_fused_rmsnorm_f32` primitive. The generated kernel computes the
Gemma-style fused RMSNorm FP32 reference:

```text
out[row, col] = x[row, col] * rsqrt(mean(x[row, :]^2) + eps) * (1.0 + weight[col])
```

for a rank-2 input tensor and a rank-1 weight vector.

## Harness

The harness is `examples/cuda/gluon_gemma_fused_rmsnorm_f32.py`. It emits
structured JSON for pass, skip, and fail cases. The payload records:

- `kernel_name`, `status`, `schema_version`, and the reference expression;
- generated artifact paths and source digest;
- `rows`, `hidden`, epsilon, and tolerance;
- maximum absolute error when correctness is measured.

The default output directory is repo-relative
`tmp/gluon-gemma-fused-rmsnorm-local`. Absolute `--output-dir` values are
rejected so stdout JSON does not leak private local paths.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_gemma_fused_rmsnorm_f32.py \
  --output-dir tmp/gluon-gemma-fused-rmsnorm-local \
  --rows 2 --hidden 16 --eps 1e-5 --arch compute_90
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `rows=2`, `hidden=16`;
- epsilon: `1e-5`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `0.0`;
- source digest:
  `30d3ca974420d95b1e9bcabfda3694c4a5f8c56dec32ea925179e5b7a8c6cbf8`.

Local GPU metadata:

- GPU: NVIDIA A100 family;
- compute capability: `8.0`;
- driver: `595.71.05`;
- CUDA toolkit: `nvcc` from CUDA 12.8.

## H200 Command

The H200 run used tree sync into `<remote-pto-cu>` and a preserved remote
Gluon Python environment. The committed command keeps those paths sanitized.

Execution details:

- synced checkout: `<remote-pto-cu>`;
- python environment: <remote-gluon-venv>;
- GPU: NVIDIA H200 NVL;
- compute capability: `9.0`;
- driver: `580.126.20`;
- CUDA toolkit: `<cuda-toolkit>`, CUDA 12.8 compiler build;
- Torch: `2.11.0+cu130`;
- Torch CUDA: `13.0`;
- Triton: `3.6.0`.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_gemma_fused_rmsnorm_f32.py \
      --output-dir tmp/gluon-gemma-fused-rmsnorm-h200 \
      --rows 2 --hidden 16 --eps 1e-5 \
      --require-cuda --device 0 --arch compute_90'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `rows=2`, `hidden=16`;
- epsilon: `1e-5`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `0.0`;
- source digest:
  `30d3ca974420d95b1e9bcabfda3694c4a5f8c56dec32ea925179e5b7a8c6cbf8`.

## Limitations

- The evidence covers one bounded FP32 Gemma-style fused RMSNorm shape.
- The implementation is correctness-focused, not performance-focused.
- The remote evidence used tree sync plus a preserved remote Gluon venv rather
  than a venv created inside the fresh synced checkout.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not production serving readiness.
- This is not DeepSeek semantic correctness.
- This is not broader normalization coverage.
- This is not activation coverage.
- This is not fused attention evidence.
- This is not KV-cache integration evidence.
- This is not throughput or latency evidence.
- This is not vLLM/simpler-nv integration evidence.

## Follow-Up Gaps

- Add broader RMSNorm and LayerNorm shape coverage in separate slices.
- Keep activation, fused attention, KV-cache, and serving integration evidence
  in their own bounded branches.
