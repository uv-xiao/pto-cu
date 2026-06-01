# 2026-06-02 Qwen VDCores Full Diagnostic Decode

## Code And Data Changed

- Added `--resource-backed-workload` so the Qwen decode-loop runner can narrow
  resource-backed execution to one serving policy.
- Added `--resource-backed-logits-check-policy final_step`, which defers the
  full logits-buffer readback until the last bounded diagnostic decode step.
- Split resource-backed result assembly into
  `examples/cuda/qwen_decode_loop_runner_impl/resource_backed_results.py` and
  kept policy helpers in
  `examples/cuda/qwen_decode_loop_runner_impl/resource_check_policy.py`.
- Imported the new raw artifact:
  `tmp/cuda-backend/pto-serving-resource-backed-vdcores-full-decode-2026-06-02/`
  `qwen-decode-loop-runner.json`.

## Architecture Quality

- The default policy remains `every_step`, preserving prior artifact behavior.
- The `final_step` policy keeps per-step device token-feedback observation,
  while avoiding repeated full-buffer host readback on non-final steps.
- Viewer rows now expose `logits_check_policy`,
  `logits_checked_step_count`, and `logits_deferred_step_count`.

## Evaluation Run

Command:

```bash
ARTIFACT_DIR=tmp/cuda-backend/pto-serving-resource-backed-vdcores-full-decode-2026-06-02
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-decode-steps 64 \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --output-json "$ARTIFACT_DIR/qwen-decode-loop-runner.json"
```

Result:

- `vdcores_offline_decode` executed all 64 planned diagnostic decode steps.
- The scheduler completed `16,320` total task executions with zero errors.
- Device-side sampled-token feedback was observed for all 64 steps.
- The final step checked all `2,430,976` diagnostic logits elements with
  max absolute error `0.0`.

## Remaining Gaps

- Task bodies still use diagnostic Qwen formulas, not numerically correct Qwen
  kernels.
- This proves VDCores policy decode-loop mechanics locally on A100; MPK needs
  the same longer-loop run after the remaining full-serving work narrows.
