# 2026-06-02 Qwen MPK 1024 Diagnostic Decode

## Code And Data Changed

- Imported a full planned MPK diagnostic decode-loop artifact for
  `mpk_offline_decode`.
- The raw artifact is:
  `tmp/cuda-backend/pto-serving-resource-backed-mpk-1024-decode-2026-06-02/`
  `qwen-decode-loop-runner.json`.
- Refreshed benchmark-viewer data, paper-readiness audit data, and goal
  progress data after import.

## Architecture Quality

This artifact uses the same resource-backed persistent-device runner and
`final_step` logits-check policy as the shorter diagnostic runs. It proves the
runner can keep scheduler state, resource bindings, and device-side diagnostic
token feedback coherent across the full 1,024 planned MPK decode steps.

## Evaluation Run

Command:

```bash
ARTIFACT_DIR=tmp/cuda-backend/pto-serving-resource-backed-mpk-1024-decode-2026-06-02
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1024 \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --output-json "$ARTIFACT_DIR/qwen-decode-loop-runner.json"
```

Result:

- `mpk_offline_decode` executed all 1,024 planned diagnostic decode steps.
- The scheduler completed `261,120` total task executions with zero errors.
- Device-side sampled-token feedback was observed for all 1,024 steps.
- The final step checked all `2,430,976` diagnostic logits elements with
  max absolute error `0.0`.

## Remaining Gaps

- Task bodies still use diagnostic Qwen formulas, not numerically correct Qwen
  kernels.
- The remaining PTO work is full numerical serving execution and paper-ready
  full-serving viewer rows.
