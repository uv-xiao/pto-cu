# Gluon RoPE FP32 H200 Correctness

This note records correctness evidence for the generated Gluon `rope_f32`
primitive. The generated kernel applies precomputed cosine and sine tensors to
adjacent even/odd feature pairs:

```text
out_even = x_even * cos - x_odd * sin
out_odd = x_even * sin + x_odd * cos
```

for bounded rank-3 input tensors.

## Harness

The harness is `examples/cuda/gluon_rope_f32.py`. It emits structured JSON for
pass, skip, and fail cases. The default path preserves the existing single
case CLI behavior:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_rope_f32.py \
  --output-dir tmp/gluon-rope-local \
  --batch 1 --seq 2 --head-dim 8 --arch compute_90
```

The sweep path emits aggregate JSON with `schema_version: 1`, aggregate
status and counts, and per-case shape, provenance, artifact metadata, status,
and max-error fields when correctness is measured. Artifact paths are
repo-relative, and private absolute paths are not recorded. Absolute
`--output-dir` values are rejected for both paths.

Sweep cases:

- existing smoke: `batch=1, seq=2, head_dim=8`;
- DeepSeek-V4-Flash RoPE representative: `batch=1, seq=4, head_dim=64`, with
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  `rope_head_dim: 64` provenance.

## H200 Command

The H200 run used tree sync into `<remote-pto-cu>` and a preserved remote
Gluon Python environment. The committed command keeps those paths sanitized.

Execution details:

- synced checkout: `<remote-pto-cu>`;
- python environment: <remote-gluon-venv>;
- machine class: H200.

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/gluon_rope_f32.py \
      --output-dir tmp/gluon-rope-shape-coverage-h200 \
      --sweep --require-cuda --device 0 --arch compute_90'
```

H200 result from stdout JSON:

- schema_version: 1;
- status: passed;
- case_count: 2;
- passed_cases: 2;
- failed_cases: 0;
- skipped_cases: 0;
- case statuses: passed, passed;
- artifact paths are repo-relative;
- private absolute paths are not recorded;
- source digest for both cases:
  `6cd4d800c0e8cb943a8e12f8f5f93f2d7df024b7f70df433a5b7d3f4616b0d53`.

Per-case results:

- `existing_smoke`: shape `batch=1, seq=2, head_dim=8`, provenance
  `existing smoke correctness fixture`, max absolute error:
  `2.384185791015625e-07`;
- `deepseek_v4_flash_rope_head_dim64`: shape
  `batch=1, seq=4, head_dim=64`, provenance
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  `rope_head_dim: 64`, max absolute error: `2.384185791015625e-07`.

## Limitations

- The evidence covers a small FP32 RoPE shape sweep only.
- The implementation is correctness-focused, not performance-focused.
- The remote evidence used tree sync plus a preserved remote Gluon venv rather
  than a venv created inside the fresh synced checkout.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not production serving readiness.
- This is not DeepSeek semantic correctness.
- This is not fused attention evidence.
- This is not KV-cache integration evidence.
- This is not batched decode or prefill integration evidence.
- This is not vLLM/simpler-nv integration evidence.
- This is not throughput or latency evidence.

## Follow-Up Gaps

- Keep fused attention, KV-cache mutation, paged attention, and serving
  integration evidence in their own bounded branches.
- Keep LayerNorm, SiLU, GELU, gated activation, and fused normalization
  fixtures in separate non-RoPE slices.
