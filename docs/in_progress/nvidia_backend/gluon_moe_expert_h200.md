# Gluon MoE Expert Affine H200 Correctness

This note records correctness evidence for the generated Gluon
`moe_expert_affine_f32` primitive. The generated kernel computes:

```text
out = scale_a * a + scale_b * b
```

for FP32 vectors.

## Harness

The harness is `examples/cuda/gluon_moe_expert_affine.py`. It emits structured
JSON for pass, skip, and fail cases. The payload records:

- `kernel_name`, `status`, `schema_version`;
- generated artifact paths and source digest;
- vector shape, scalar coefficients, and tolerance;
- maximum absolute error when correctness is measured.

The default output directory is repo-relative
`tmp/gluon-moe-expert-local`. Absolute `--output-dir` values are rejected so
stdout JSON does not leak private local paths. The harness suppresses local
PyTorch NumPy ABI warning noise during skip checks.

The optional `--sweep` mode runs a fixed four-case correctness sweep with
varied vector lengths, scalar coefficients, and seeds. The sweep keeps the
single-case command compatible and emits an aggregate JSON payload with
per-case artifact paths and source digests.

## Local Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_moe_expert_affine.py \
  --output-dir tmp/gluon-moe-expert-local \
  --n 4096 --arch compute_90
```

Local result:

- exit code: `0`;
- status: `passed`;
- shape: `n=4096`;
- scalars: `scale_a=1.25`, `scale_b=0.5`;
- tolerance: `atol=1e-6`, `rtol=1e-6`;
- max absolute error: `4.76837158203125e-07`;
- source digest:
  `38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842`.

## Local Sweep Command

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/gluon_moe_expert_affine.py \
  --output-dir tmp/gluon-moe-expert-sweep-local \
  --arch compute_90 --sweep
```

Local sweep result:

- exit code: `0`;
- status: `passed`;
- case count: `4`;
- passed cases: `4`;
- failed cases: `0`;
- skipped cases: `0`;
- cases:
  - `n16_baseline`: `n=16`, `scale_a=1.25`, `scale_b=0.5`,
    `seed=0`, `max_abs_error=1.1920928955078125e-07`;
  - `n31_signed`: `n=31`, `scale_a=-0.75`, `scale_b=2.0`,
    `seed=7`, `max_abs_error=0.0`;
  - `n256_b_only`: `n=256`, `scale_a=0.0`, `scale_b=-1.0`,
    `seed=13`, `max_abs_error=0.0`;
  - `n4096_mixed`: `n=4096`, `scale_a=1.5`, `scale_b=-0.25`,
    `seed=23`, `max_abs_error=4.76837158203125e-07`;
- source digest for all cases:
  `38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842`.

## H200 Command

The fresh synced remote path did not contain `.venv`. Per the worker prompt,
the run used the preserved H200 Gluon venv from an earlier Gluon worker:
`/tmp/pto-cu-gluon-gemm/.venv/bin/python`. That venv provided Torch
`2.8.0+cu128` and Triton `3.7.1` on an NVIDIA H200 NVL.

```bash
set -o pipefail; \
REMOTE_PTO_CU=/tmp/pto-cu-gluon-moe-expert \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    /tmp/pto-cu-gluon-gemm/.venv/bin/python \
    examples/cuda/gluon_moe_expert_affine.py \
    --output-dir tmp/gluon-moe-expert-h200 \
    --n 4096 --arch compute_90 --require-cuda' \
  | tee tmp/gluon-moe-expert-h200-remote.json
```

H200 result:

- exit code: `0`;
- status: `passed`;
- shape: `n=4096`;
- scalars: `scale_a=1.25`, `scale_b=0.5`;
- tolerance: `atol=1e-6`, `rtol=1e-6`;
- max absolute error: `4.76837158203125e-07`;
- source digest:
  `38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842`;
- raw local capture: `tmp/gluon-moe-expert-h200-remote.json`.

## H200 Sweep Command

The run used `--sync` into `/tmp/pto-cu-gluon-moe-expert-sweep` and the same
preserved H200 Gluon venv:
`/tmp/pto-cu-gluon-gemm/.venv/bin/python`. The H200 host reported NVIDIA H200
NVL GPUs with compute capability `9.0`, driver `580.126.20`, and CUDA toolkit
`/usr/local/cuda` (`nvcc` 12.8). The Python environment reported Torch
`2.8.0+cu128`, Torch CUDA `12.8`, and Triton `3.7.1`.

```bash
REMOTE_PTO_CU=/tmp/pto-cu-gluon-moe-expert-sweep \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    /tmp/pto-cu-gluon-gemm/.venv/bin/python \
    examples/cuda/gluon_moe_expert_affine.py \
    --output-dir tmp/gluon-moe-expert-sweep-h200 \
    --arch compute_90 --require-cuda --sweep'
```

H200 sweep result:

- exit code: `0`;
- status: `passed`;
- case count: `4`;
- passed cases: `4`;
- failed cases: `0`;
- skipped cases: `0`;
- cases:
  - `n16_baseline`: `n=16`, `scale_a=1.25`, `scale_b=0.5`,
    `seed=0`, `max_abs_error=1.1920928955078125e-07`;
  - `n31_signed`: `n=31`, `scale_a=-0.75`, `scale_b=2.0`,
    `seed=7`, `max_abs_error=0.0`;
  - `n256_b_only`: `n=256`, `scale_a=0.0`, `scale_b=-1.0`,
    `seed=13`, `max_abs_error=0.0`;
  - `n4096_mixed`: `n=4096`, `scale_a=1.5`, `scale_b=-0.25`,
    `seed=23`, `max_abs_error=4.76837158203125e-07`;
- source digest for all cases:
  `38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842`.

## Limitations

- The single-case evidence covers one FP32 vector shape and two scalar
  coefficients.
- The sweep evidence covers four FP32 vector shapes and scalar coefficient
  pairs.
- This does not benchmark performance.
- The remote evidence used tree sync plus a preserved remote venv rather than
  a venv created in the fresh synced path.

## Non-Claims

- No persistent-device integration is claimed.
- No fused MoE dispatch/combine behavior is claimed.
- No distributed behavior, UCCL/NCCL integration, serving behavior,
  DeepSeek behavior, or performance is claimed.

## Follow-Up Gaps

- Add a persistent-device integration slice only after this generated primitive
  remains stable.
- Keep fused dispatch/combine, multi-GPU, and serving validation in their own
  evidence-producing branches.
