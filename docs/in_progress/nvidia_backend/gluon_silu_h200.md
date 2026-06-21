# Gluon SiLU FP32 H200 Correctness

This note records correctness evidence for the generated Gluon `silu_f32`
primitive. The generated kernel computes the FP32 SiLU reference:

```text
out = x / (1.0 + exp(-x))
```

for bounded rank-1 input tensors. This is equivalent to
`out = x * sigmoid(x)`. This is standalone SiLU activation correctness
evidence only.

## Harness

The harness is `examples/cuda/gluon_silu_f32.py`. It emits structured JSON for
pass, skip, and fail cases. The default command still runs one case with
`--n 32`. The `--sweep` path emits aggregate JSON with:

- `schema_version: 1`, aggregate `status`, and pass/fail/skip counts;
- per-case shape, provenance, reference formula, tolerance, and status;
- generated artifact paths and source digest;
- maximum absolute error when correctness is measured.

The default output directory is repo-relative `tmp/gluon-silu-local`.
Absolute `--output-dir` values are rejected for both the default path and the
sweep path so stdout JSON does not leak private local paths. Command display
also sanitizes equals-form absolute path arguments such as
`--output-dir=/tmp/private-output`.

## Sweep Cases

- Existing smoke fixture: `n=32`, provenance
  `existing smoke correctness fixture`.
- DeepSeek-V4-Flash representative standalone SiLU width: `n=2048`, with
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  `moe_inter_dim: 2048` and `swiglu_limit: 10.0` provenance.

The `n=2048` case passed on H200, so no smaller passing boundary was needed.

## H200 Command

The H200 run used tree sync into `<remote-pto-cu>` and a preserved remote
Gluon Python environment. The committed command keeps those paths sanitized.

Execution details:

- synced checkout: `<remote-pto-cu>`;
- source commit before this branch commit: `8a0bf6cdc7b2929910f1574005b806508ef40986`
  plus the local SiLU sweep diff synced by the wrapper;
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
      examples/cuda/gluon_silu_f32.py \
      --output-dir tmp/gluon-silu-shape-coverage-h200 \
      --sweep --require-cuda --device 0 --arch compute_90'
```

H200 aggregate result:

- exit code: `0`;
- status: passed;
- case count: `2`;
- passed cases: `2`;
- failed cases: `0`;
- skipped cases: `0`;
- max absolute error: `4.76837158203125e-07` across the sweep;
- source digest:
  `760590d7df8971d35dc8885be5aeeb2b4a7cf2cb4340ae604a7d7df84ad31913`.

Per-case H200 result:

| Case | Shape | Status | Max absolute error |
| --- | --- | --- | --- |
| existing smoke | `n=32` | passed | `1.1920928955078125e-07` |
| DeepSeek-V4-Flash representative | `n=2048` | passed | `4.76837158203125e-07` |

## Limitations

- The evidence covers two bounded FP32 SiLU vector shapes.
- The `n=2048` case is standalone SiLU gate-activation-width evidence tied to
  DeepSeek-V4-Flash config fields; it does not execute a gated activation
  kernel or model graph.
- The implementation is correctness-focused, not performance-focused.
- The remote evidence used tree sync plus a preserved remote Gluon venv rather
  than a venv created inside the fresh synced checkout.

## Non-Claims

- This is not FlashInfer integration evidence.
- This is not production serving readiness.
- This is not DeepSeek semantic correctness.
- This is not gated activation coverage.
- This is not GELU coverage.
- This is not broader activation coverage.
- This is not Gemma-style fused norm coverage.
- This is not fused attention evidence.
- This is not KV-cache integration evidence.
- This is not vLLM/simpler-nv integration evidence.
- This is not throughput or latency evidence.

## Follow-Up Gaps

- Keep gated activation, GELU, broader activation, fused attention, KV-cache
  mutation, paged attention, and serving integration evidence in their own
  bounded branches.
