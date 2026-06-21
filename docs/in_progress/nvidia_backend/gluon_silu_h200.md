# Gluon SiLU FP32 H200 Correctness

This note records correctness evidence for the generated Gluon `silu_f32`
primitive. The generated kernel computes the FP32 SiLU reference:

```text
out = x * sigmoid(x)
```

for a bounded rank-1 input tensor.

## Harness

The harness is `examples/cuda/gluon_silu_f32.py`. It emits structured JSON for
pass, skip, and fail cases. The payload records:

- `kernel_name`, `status`, and `schema_version`;
- generated artifact paths and source digest;
- vector length and tolerance;
- maximum absolute error when correctness is measured.

The default output directory is repo-relative `tmp/gluon-silu-local`.
Absolute `--output-dir` values are rejected so stdout JSON does not leak
private local paths. The harness suppresses local PyTorch NumPy ABI warning
noise during skip checks.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_silu_f32.py \
  --output-dir tmp/gluon-silu-local \
  --n 32 --arch compute_90
```

Local result:

- exit code: `0`;
- status: passed;
- shape: `n=32`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `1.1920928955078125e-07`;
- source digest:
  `760590d7df8971d35dc8885be5aeeb2b4a7cf2cb4340ae604a7d7df84ad31913`.

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
- Torch: `2.8.0+cu128`;
- Torch CUDA: `12.8`;
- Triton: `3.7.1`.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_silu_f32.py \
      --output-dir tmp/gluon-silu-h200 \
      --n 32 \
      --require-cuda --device 0 --arch compute_90'
```

H200 result:

- exit code: `0`;
- status: passed;
- shape: `n=32`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `1.1920928955078125e-07`;
- source digest:
  `760590d7df8971d35dc8885be5aeeb2b4a7cf2cb4340ae604a7d7df84ad31913`.

## Limitations

- The evidence covers one bounded FP32 SiLU vector shape.
- The implementation is correctness-focused, not performance-focused.
- The remote evidence used tree sync plus a preserved remote Gluon venv rather
  than a venv created inside the fresh synced checkout.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not production serving readiness.
- This is not DeepSeek semantic correctness.
- This is not GELU coverage.
- This is not gated activation coverage.
- This is not Gemma-style fused norm coverage.
- This is not fused attention evidence.
- This is not KV-cache integration evidence.
- This is not vLLM/simpler-nv integration evidence.
- This is not throughput or latency evidence.

## Follow-Up Gaps

- Add GELU and gated activation fixtures in separate activation slices.
- Keep Gemma-style fused norm, fused attention, KV-cache mutation, paged
  attention, and serving integration evidence in their own bounded branches.
