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

## Limitations

- This covers one FP32 vector shape and two scalar coefficients.
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
- Add broader shape and coefficient coverage in a separate PR-sized slice.
- Keep fused dispatch/combine, multi-GPU, and serving validation in their own
  evidence-producing branches.
