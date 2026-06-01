# 2026-06-01 Qwen Decode Loop Runner Plan

## Code And Data Changed

- Added `examples/cuda/qwen_decode_loop_runner.py` plus
  `examples/cuda/qwen_decode_loop_runner_impl/`.
- The artifact composes token pointer, KV-cache, and resident-weight lifecycle
  artifacts into persistent DAG submission plans for the MPK and VDCores
  serving policies.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Added a runner-owned unit-math live bridge so the decode-loop runner entry
  point can execute the repeated diagnostic unit-math DAG.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-decode-loop-2026-06-01/`
  `qwen-decode-loop-runner.json`.

## Architecture Quality

The runner artifact makes the host-side lifetime boundary explicit:
token pointers, KV-cache pointers, and resident weight pointers are opened
before decode and weight argument materialization, remain live through
persistent DAG submission, and close after submission. It also records output
token accounting for each serving policy. The unit-math bridge keeps the
live CUDA evidence diagnostic, but proves the runner can attach an executed
CUDA sub-DAG summary instead of only pointing at adjacent artifacts.

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

Additional bridge command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --run-unit-math-live --device 0 --arch compute_80 \
  --repeat-runs 3 --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-unit-math-bridge-2026-06-01/\
qwen-decode-loop-runner.json
```

Result: `unit_math_live_bridge_contract.status=diagnostic_bridge_executed`,
`repeat_runs=3`, `total_completed_count=12`, `total_error_count=0`, and
`max_abs_error=0.0`.

## Remaining Gaps

- Generate Qwen kernel bodies that consume token, KV-cache, and weight fields.
- Replace the dry-run plan with a `cuda_live` decode loop and import
  full-serving viewer rows.
