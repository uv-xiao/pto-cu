# 2026-06-01 Qwen Decode Loop Runner Plan

## Code And Data Changed

- Added `examples/cuda/qwen_decode_loop_runner.py` plus
  `examples/cuda/qwen_decode_loop_runner_impl/`.
- The artifact composes token pointer, KV-cache, and resident-weight lifecycle
  artifacts into persistent DAG submission plans for the MPK and VDCores
  serving policies.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-decode-loop-2026-06-01/`
  `qwen-decode-loop-runner.json`.

## Architecture Quality

The runner artifact makes the host-side lifetime boundary explicit:
token pointers, KV-cache pointers, and resident weight pointers are opened
before decode and weight argument materialization, remain live through
persistent DAG submission, and close after submission. It also records output
token accounting for each serving policy.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline \
  --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-2026-06-01/\
qwen-decode-loop-runner.json
```

Result: `status=decode_loop_runner_plan_ready` with 1088 planned decode
iterations across `mpk_offline_decode` and `vdcores_offline_decode`.

## Remaining Gaps

- Generate Qwen kernel bodies that consume token, KV-cache, and weight fields.
- Replace the dry-run plan with a `cuda_live` decode loop and import
  full-serving viewer rows.
