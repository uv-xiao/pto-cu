# 2026-06-02 Qwen VDCores Weighted Elementwise Bridge

## Code And Data Changed

- Expanded resource-backed `unit_math` coverage from the RMSNorm/QKV/MLP
  bridge to eight Qwen task-body callables.
- Added weighted elementwise coverage metadata for `qwen_attention_qk_norm`,
  `qwen_attention_o`, `qwen_rmsnorm_post_attention`, `qwen_mlp_down`, and
  `qwen_final_norm`.
- Imported the VDCores serving-policy artifact at
  `$ARTIFACT/qwen-decode-loop-runner.json`.

## Architecture Quality

The artifact separates safe O(n) weighted elementwise task bodies from the
remaining full-serving gaps. Embedding and logits are intentionally not
promoted as unit-math callables: embedding still needs a token-safe full
hidden-state mapping, and logits still uses the diagnostic full-buffer
formula.

## Evaluation Run

Command:

```bash
ARTIFACT=tmp/cuda-backend/pto-serving-resource-backed-vdcores-weighted-elementwise
ARTIFACT=$ARTIFACT-2026-06-02
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
- The artifact records 8 numeric-ready callables and 5 weighted elementwise
  callables.
- The final step checked all `2,430,976` diagnostic logits elements with
  max absolute error `0.0`.

## Remaining Gaps

- Implement token-safe embedding and full logits matmul instead of the
  diagnostic logits formula.
- Replace remaining bridge formulas with full hidden-size Qwen kernels before
  importing PTO full-serving paper rows.
