# 2026-06-04 Qwen Layer-2 Full-Row Stages

## Code And Data Changed

- Fixed activation finiteness summaries to include trailing activation tasks
  when the final packet task is not `qwen_logits`.
- Added a focused regression for that selector so `qwen_mlp_down` is not
  dropped merely because it is last in a prefill packet.
- Captured layer-2 full rows for the layer-3 MPK model-equivalent boundary:
  input norm, QKV, RoPE-applied Q/K, attention-O, post-attention norm,
  MLP activation product, and MLP-down output.

## Architecture Quality

This keeps the evidence path on benchmark-model execution. The code change is
not a new test layer or viewer contract; it removes a diagnostic blind spot
that hid the last activation-producing task in packets without logits.

The full-row comparison rules out a gross layer-2 MLP-down handoff failure.
`layer_2_mlp_down` is close to Hugging Face across the full row, while the
largest distributed substage drift is in RoPE-applied Q/K. That points the
next root-cause slice toward precision and accumulation behavior around QK/RoPE
and final normalization.

## Evaluation Run

PTO probe:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-prefill-prompt \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 3 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 512 \
  --resource-backed-projection-active-cols full \
  --resource-backed-numeric-task-mode model_equivalent \
  --resource-backed-activation-row-dump-descriptor-ids \
  layer_2_input_norm,layer_2_attention_qkv,layer_2_attention_qk_norm,layer_2_attention_o,layer_2_post_attention_norm,layer_2_mlp_gate_up,layer_2_mlp_down
```

Artifacts:

- Artifact root:
  `tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-model-equivalent-stage-full-rows-v2/`
- `qwen-runner.json`
- `pto-hf-layer2-stage-full-row-comparison.json`

Results:

- The PTO runner reports `resource_backed_execution.status=pass`.
- The final prefill packet now dumps `layer_2_mlp_down.row_values` with 4,096
  columns, proving the trailing activation diagnostic gap is closed.
- First-512 PTO top-k remains `[10, 167, 376, 475, 58]`.
- Full-row stage comparison against Hugging Face reports:
  `layer_2_input_norm.mean_abs_delta=0.000093`,
  `layer_2_attention_qkv.mean_abs_delta=0.000239`,
  `layer_2_attention_qk_norm.mean_abs_delta=0.005142`,
  `layer_2_attention_o.mean_abs_delta=0.000497`,
  `layer_2_post_attention_norm.mean_abs_delta=0.001962`,
  `layer_2_mlp_gate_up.mean_abs_delta=0.000055`, and
  `layer_2_mlp_down.mean_abs_delta=0.000989`.

## Remaining Gaps

Full Qwen correctness remains open. The next benchmark-model slice should
test whether QK/RoPE precision or final RMSNorm accumulation can explain the
remaining token-229-minus-token-58 boundary drift.
