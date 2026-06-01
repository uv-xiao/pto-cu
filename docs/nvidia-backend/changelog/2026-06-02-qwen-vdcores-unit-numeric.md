# 2026-06-02 Qwen VDCores Unit Numeric Branches

## Code And Data Changed

- Added `--resource-backed-numeric-task-mode unit_math` to the Qwen
  decode-loop runner.
- The opt-in mode enables the safe O(n) resource-backed task-body branches
  for `qwen_attention_qkv` and `qwen_mlp_gate_up`.
- Imported the VDCores serving-policy artifact:
  `tmp/cuda-backend/pto-serving-resource-backed-vdcores-unit-numeric-2026-06-02/`
  `qwen-decode-loop-runner.json`.

## Architecture Quality

The mode is explicit in the raw artifact, benchmark-viewer statistics, and
paper-evaluation matrix symbols. It does not claim full Qwen correctness:
RMSNorm remains covered by the small unit-math live DAG, because the current
resource-backed RMSNorm branch is not suitable for full hidden buffers.

## Evaluation Run

Command:

```bash
ARTIFACT_DIR=tmp/cuda-backend/pto-serving-resource-backed-vdcores-unit-numeric-2026-06-02
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-decode-steps 64 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math \
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

- This is still a partial numeric-mode bridge, not full Qwen serving
  correctness.
- Remaining PTO work is numerically correct full hidden-size kernels,
  full-serving decode execution, and paper-ready full-serving viewer rows.
