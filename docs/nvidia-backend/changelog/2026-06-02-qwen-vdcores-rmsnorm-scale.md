# 2026-06-02 Qwen VDCores RMSNorm Scale Bridge

## Code And Data Changed

- Added an O(n) external-scale path to the generated `qwen_rmsnorm_input`
  persistent-device task body.
- Expanded `--resource-backed-numeric-task-mode unit_math` to mark
  `qwen_rmsnorm_input`, `qwen_attention_qkv`, and `qwen_mlp_gate_up` as
  numeric-ready resource-backed callables.
- Imported the VDCores serving-policy artifact:
  `tmp/cuda-backend/pto-serving-resource-backed-vdcores-rmsnorm-scale-2026-06-02/`
  `qwen-decode-loop-runner.json`.

## Architecture Quality

The RMSNorm path records an explicit external scale contract through
`scalar_args[1]`. This keeps the full hidden buffer branch O(n) for
resource-backed execution while preserving the small hidden4 reduction path in
unit tests. It is a bridge toward full Qwen RMSNorm, not final numerical
serving correctness.

## Evaluation Run

The VDCores resource-backed row executed 64 bounded decode steps with
`--resource-backed-logits-check-policy final_step` and
`--resource-backed-numeric-task-mode unit_math`.

Command:

```bash
ARTIFACT=tmp/cuda-backend/pto-serving-resource-backed-vdcores-rmsnorm-scale-2026-06-02
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-decode-steps 64 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math \
  --device 0 --arch compute_80 \
  --output-json "$ARTIFACT/qwen-decode-loop-runner.json"
```

Result:

- `vdcores_offline_decode` completed all 64 planned diagnostic decode steps.
- The scheduler completed `16,320` total task executions with zero errors.
- Device-side sampled-token feedback was observed for all 64 steps.
- The final step checked all `2,430,976` diagnostic logits elements with
  max absolute error `0.0`.

## Remaining Gaps

- Replace the external-scale RMSNorm bridge with full Qwen RMSNorm reduction
  and hidden-size kernels.
- Import PTO full-serving rows only after the decode loop is numerically
  correct beyond diagnostic resource-backed formulas.
