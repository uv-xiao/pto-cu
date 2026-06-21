# Gluon Gated SiLU FP32 H200 Correctness

This note records correctness evidence for the generated Gluon
`gated_silu_f32` primitive. The generated kernel computes the bounded FP32
gated SiLU reference:

```text
out = value * gate / (1.0 + exp(-gate))
```

for rank-1 `gate` and `value` tensors.

## Harness

The harness is `examples/cuda/gluon_gated_silu_f32.py`. It emits structured
JSON for pass, skip, and fail cases. The default path remains a single
`--n 32` correctness check. The optional `--sweep` path emits aggregate JSON
with `schema_version: 1`, aggregate status, case counts, and per-case shape,
provenance, artifact metadata, reference formula, status, and maximum
absolute error when correctness is measured.

The sweep cases are:

- existing smoke fixture, shape: `n=32`;
- DeepSeek-V4-Flash representative gated activation width, shape: `n=2048`,
  with provenance from
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  fields `moe_inter_dim: 2048` and `swiglu_limit: 10.0`.

The default output directory is repo-relative `tmp/gluon-gated-silu-local`.
Absolute `--output-dir` values are rejected in both the default and sweep
paths so stdout JSON does not leak private local paths. The harness checks
that Gluon exposes `gl.exp`; if it is missing, the harness reports a
structured skip instead of using an approximation.

## Local Command

Single-case default path:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_gated_silu_f32.py \
  --output-dir tmp/gluon-gated-silu-local \
  --n 32 --arch compute_90
```

Shape sweep path:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_gated_silu_f32.py \
  --output-dir tmp/gluon-gated-silu-shape-coverage-local \
  --sweep --arch compute_90
```

Local result from the pre-existing single case:

- exit code: `0`;
- status: passed;
- shape: `n=32`;
- reference: `out = value * gate / (1.0 + exp(-gate))`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `4.76837158203125e-07`;
- source digest:
  `5217e47b2c2db4a67ecb81ba690cc069e32956f048b6061457409a2b77b032de`.

Local GPU metadata:

- GPU: NVIDIA A100 family;
- compute capability: `8.0`;
- driver: `595.71.05`;
- CUDA toolkit: `nvcc` from CUDA 12.8;
- Torch: `2.1.0+cu121`;
- Torch CUDA: `12.1`;
- Triton: `3.5.1`;
- Gluon `gl.exp`: available.

## H200 Command

The H200 run used tree sync into `<remote-pto-cu>` and a preserved remote
Python environment with Gluon available. The committed command keeps those
paths sanitized.

Execution details:

- synced checkout: `<remote-pto-cu>`;
- python environment: <remote-gluon-venv>;
- GPU: NVIDIA H200 NVL;
- compute capability: `9.0`;
- memory: `143771 MiB`;
- driver: `580.126.20`;
- CUDA toolkit: `<cuda-toolkit>`, CUDA 12.8 compiler build;
- Torch: `2.11.0+cu130`;
- Torch CUDA: `13.0`;
- Triton: `3.6.0`;
- Gluon `gl.exp`: available.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_gated_silu_f32.py \
      --output-dir tmp/gluon-gated-silu-shape-coverage-h200 \
      --sweep --require-cuda --device 0 --arch compute_90'
```

H200 result:

- exit code: `0`;
- status: passed;
- case count: `2`;
- passed cases: `2`;
- failed cases: `0`;
- skipped cases: `0`;
- reference: `out = value * gate / (1.0 + exp(-gate))`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- source digest:
  `5217e47b2c2db4a67ecb81ba690cc069e32956f048b6061457409a2b77b032de`.

Per-case result:

- `existing_smoke`: shape: `n=32`, status: passed, max absolute error:
  `4.76837158203125e-07`.
- `deepseek_v4_flash_moe_inter_dim`: shape: `n=2048`, status: passed,
  provenance:
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  `moe_inter_dim: 2048`, `swiglu_limit: 10.0`, max absolute error:
  `9.5367431640625e-07`.

## Limitations

- The evidence covers a small bounded FP32 gated SiLU vector shape sweep.
- The implementation is correctness-focused, not performance-focused.
- The remote evidence used tree sync plus a preserved remote Python
  environment rather than a venv created inside the fresh synced checkout.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not production serving readiness.
- This is not DeepSeek semantic correctness.
- This is not Gemma-style fused norm coverage.
- This is not broader activation coverage.
- This is not fused attention evidence.
- This is not KV-cache integration evidence.
- This is not vLLM/simpler-nv integration evidence.
- This is not throughput or latency evidence.

## Follow-Up Gaps

- Keep FlashInfer integration, serving readiness, fused attention, KV-cache
  mutation, paged attention, broader activation coverage, and serving
  integration evidence in their own bounded branches.
