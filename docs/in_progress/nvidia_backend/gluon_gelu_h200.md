# Gluon GELU FP32 H200 Correctness

This note records correctness evidence for the generated Gluon `gelu_f32`
primitive. The generated kernel computes the exact FP32 GELU reference:

```text
0.5 * x * (1.0 + erf(x / sqrt(2.0)))
```

for a bounded rank-1 input tensor.

## Harness

The harness is `examples/cuda/gluon_gelu_f32.py`. It emits structured JSON for
pass, skip, and fail cases. The payload records:

- `kernel_name`, `status`, and `schema_version`;
- generated artifact paths and source digest;
- vector length, reference expression, and tolerance;
- maximum absolute error when correctness is measured.

The default output directory is repo-relative `tmp/gluon-gelu-local`.
Absolute `--output-dir` values are rejected so stdout JSON does not leak
private local paths. The harness checks that Gluon exposes `gl.erf`; if it is
missing, the harness reports a structured skip instead of using an
approximation.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_gelu_f32.py \
  --output-dir tmp/gluon-gelu-local \
  --n 32 --arch compute_90
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `n=32`;
- reference: `0.5 * x * (1.0 + erf(x / sqrt(2.0)))`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `2.384185791015625e-07`;
- source digest:
  `b2ea8a4ee3ea3a9599db17d63e8bdc41f104fcdef5895476a2d170ea6d3db94a`.

Local GPU metadata:

- GPU: NVIDIA A100 80GB PCIe;
- compute capability: `8.0`;
- driver: `595.71.05`;
- CUDA toolkit: `nvcc` from CUDA 12.8;
- Torch: `2.1.0+cu121`;
- Torch CUDA: `12.1`;
- Triton: `3.5.1`.

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
- Torch: `2.8.0+cu128`;
- Torch CUDA: `12.8`;
- Triton: `3.7.1`;
- Gluon `gl.erf`: available.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_gelu_f32.py \
      --output-dir tmp/gluon-gelu-h200 \
      --n 32 \
      --require-cuda --device 0 --arch compute_90'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `n=32`;
- reference: `0.5 * x * (1.0 + erf(x / sqrt(2.0)))`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `2.384185791015625e-07`;
- source digest:
  `b2ea8a4ee3ea3a9599db17d63e8bdc41f104fcdef5895476a2d170ea6d3db94a`.

## Limitations

- The evidence covers one bounded FP32 GELU vector shape.
- The implementation is correctness-focused, not performance-focused.
- The remote evidence used tree sync plus a preserved remote Gluon venv rather
  than a venv created inside the fresh synced checkout.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not production serving readiness.
- This is not DeepSeek semantic correctness.
- This is not gated activation coverage.
- This is not Gemma-style fused norm coverage.
- This is not fused attention evidence.
- This is not KV-cache integration evidence.
- This is not vLLM/simpler-nv integration evidence.
- This is not throughput or latency evidence.

## Follow-Up Gaps

- Add gated activation fixtures in a separate activation slice.
- Keep Gemma-style fused norm, fused attention, KV-cache mutation, paged
  attention, and serving integration evidence in their own bounded branches.
