# 2026-06-02 Qwen MPK Full Diagnostic Decode

## Code And Data Changed

- Imported the MPK serving-policy counterpart to the VDCores 64-step
  resource-backed diagnostic decode artifact.
- The raw artifact is:
  `tmp/cuda-backend/pto-serving-resource-backed-mpk-full-decode-2026-06-02/`
  `qwen-decode-loop-runner.json`.
- Refreshed benchmark-viewer data, paper-readiness audit data, and goal
  progress data after import.

## Architecture Quality

The MPK and VDCores resource-backed diagnostic rows now use the same
`final_step` logits-check policy for longer bounded decode runs. This keeps
per-step scheduler and device-token-feedback evidence, while avoiding repeated
full logits-buffer host readback on non-final steps.

## Evaluation Run

Command:

```bash
ARTIFACT_DIR=tmp/cuda-backend/pto-serving-resource-backed-mpk-full-decode-2026-06-02
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 64 \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --output-json "$ARTIFACT_DIR/qwen-decode-loop-runner.json"
```

Result:

- `mpk_offline_decode` executed 64 bounded diagnostic decode steps from its
  planned 1,024-step policy.
- The scheduler completed `16,320` total task executions with zero errors.
- Device-side sampled-token feedback was observed for all 64 steps.
- The final step checked all `2,430,976` diagnostic logits elements with
  max absolute error `0.0`.

## Remaining Gaps

- Task bodies still use diagnostic Qwen formulas, not numerically correct Qwen
  kernels.
- MPK and VDCores both now have longer diagnostic decode-loop artifacts; the
  remaining PTO work is full numerical serving execution and paper-ready
  full-serving viewer rows.
