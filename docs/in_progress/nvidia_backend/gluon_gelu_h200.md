# Gluon GELU FP32 H200 Correctness

This note records correctness evidence for the generated Gluon `gelu_f32`
primitive. The generated kernel computes the exact FP32 GELU reference:

```text
0.5 * x * (1.0 + erf(x / sqrt(2.0)))
```

for bounded rank-1 input tensors.

## Harness

The harness is `examples/cuda/gluon_gelu_f32.py`. It emits structured JSON for
pass, skip, and fail cases. The payload records:

- `kernel_name`, `status`, `schema_version`, and aggregate case counts;
- generated artifact paths and source digest for each case;
- vector length, provenance, reference expression, and tolerance;
- maximum absolute error when correctness is measured.

The default output directory is repo-relative `tmp/gluon-gelu-local`. The
`--sweep` path uses a small fixed review sweep:

- `existing_smoke`: `n=32`, preserving the existing smoke correctness fixture;
- `deepseek_v4_flash_moe_inter_dim`: `n=2048`, with provenance from
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  fields `moe_inter_dim: 2048` and `swiglu_limit: 10.0` as standalone GELU
  activation-width evidence.

Absolute `--output-dir` values are rejected in both default and sweep modes so
stdout JSON does not leak private local paths. The command display also
sanitizes equals-form absolute path arguments such as
`--output-dir=/tmp/private-output`. The harness checks that Gluon exposes
`gl.erf`; if it is missing, the harness reports a structured skip instead of
using an approximation.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_gelu_f32.py \
  --output-dir tmp/gluon-gelu-local \
  --n 32 --arch compute_90
```

The default command preserves the single-case CLI path:

- shape: `n=32`;
- reference: `0.5 * x * (1.0 + erf(x / sqrt(2.0)))`;
- tolerance: `atol=1e-5`, `rtol=1e-5`.

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
- Triton: `3.6.0`;
- Gluon `gl.erf`: available.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_gelu_f32.py \
      --output-dir tmp/gluon-gelu-shape-coverage-h200 \
      --sweep --require-cuda --device 0 --arch compute_90'
```

H200 result:

- exit code: `0`;
- status: passed;
- case count: `2`;
- passed cases: `2`;
- failed cases: `0`;
- skipped cases: `0`;
- reference: `0.5 * x * (1.0 + erf(x / sqrt(2.0)))`;
- tolerance: `atol=1e-5`, `rtol=1e-5`;
- max absolute error: `2.384185791015625e-07` across both cases;
- source digest:
  `b2ea8a4ee3ea3a9599db17d63e8bdc41f104fcdef5895476a2d170ea6d3db94a`.

Case details:

- `existing_smoke`: shape `n=32`, provenance
  `existing smoke correctness fixture`, max absolute error
  `2.384185791015625e-07`.
- `deepseek_v4_flash_moe_inter_dim`: shape `n=2048`, provenance
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  `moe_inter_dim: 2048`, `swiglu_limit: 10.0`, max absolute error
  `2.384185791015625e-07`.

Artifact paths were repo-relative under
`tmp/gluon-gelu-shape-coverage-h200/<case-name>/`.

## Limitations

- The evidence covers two bounded FP32 GELU vector shapes.
- The `n=2048` case is standalone activation-width evidence tied to
  DeepSeek-V4-Flash config fields, not model execution evidence.
- The implementation is correctness-focused, not performance-focused.
- The remote evidence used tree sync plus a preserved remote Gluon venv rather
  than a venv created inside the fresh synced checkout.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not production serving readiness.
- This is not DeepSeek semantic correctness.
- This is not SiLU coverage.
- This is not gated activation coverage.
- This is not broader activation coverage.
- This is not Gemma-style fused norm coverage.
- This is not fused attention evidence.
- This is not KV-cache integration evidence.
- This is not vLLM/simpler-nv integration evidence.
- This is not throughput or latency evidence.

## Follow-Up Gaps

- Broader activation fixtures, fused attention, KV-cache mutation, paged
  attention, serving integration, and throughput or latency evidence remain
  separate from this bounded GELU correctness slice.
