# 2026-06-04 Qwen QK BF16 Boundary

## Code And Data Changed

- Added `pto_cuda_round_to_bf16_f32()` to generated CUDA persistent-DAG
  sources.
- Updated `qwen_attention_qk_norm` so model-equivalent Q/K RMSNorm and
  RoPE outputs follow the Hugging Face bf16 tensor boundary.
- Captured layer-0, layer-1, and layer-3 MPK-policy full-row probes to
  localize and measure the QK drift.

## Architecture Quality

The fix is in the Qwen task body that first diverged from Hugging Face, not in
the viewer or a row-specific assertion. Layer-0 evidence showed embedding,
input RMSNorm, and QKV projection were already close, while Q/K RMSNorm was the
first large-distribution drift. A Python reconstruction from PTO QKV matched
the CUDA QK task within `~1e-5`, which ruled out indexing and RoPE pairing
bugs and pointed to dtype-boundary mismatch.

## Evaluation Run

Focused source contract:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: passed (`10 passed`).

Layer-0 post-fix probe:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-prefill-prompt \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 1 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 512 \
  --resource-backed-projection-active-cols full \
  --resource-backed-numeric-task-mode model_equivalent \
  --resource-backed-activation-row-dump-descriptor-ids \
  layer_0_input_norm,layer_0_attention_qkv,layer_0_attention_qk_norm,layer_0_attention_o,layer_0_post_attention_norm,layer_0_mlp_gate_up,layer_0_mlp_down
```

Artifact:
`tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-qk-bf16-boundary/`.

Result: layer-0 QK mean drift improved from `0.003065` to `0.002478`;
QK p99 improved from `0.023451` to `0.018789`.

Layer-3 boundary probe:

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
  layer_2_input_norm,layer_2_attention_qkv,layer_2_attention_qk_norm,layer_2_attention_o,layer_2_post_attention_norm,layer_2_mlp_gate_up,layer_2_mlp_down,final_norm
```

Artifact:
`tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-qk-bf16-boundary/`.

Result: first-512 top-k remains `[10, 167, 376, 475, 58]`. Stage drift improves
for `layer_2_attention_qk_norm.mean_abs_delta` (`0.005142` to `0.004693`),
`layer_2_attention_o.mean_abs_delta` (`0.000497` to `0.000414`),
`layer_2_post_attention_norm.mean_abs_delta` (`0.001962` to `0.001792`), and
`layer_2_mlp_down.mean_abs_delta` (`0.000989` to `0.000847`).

## Remaining Gaps

This narrows distributed QK drift but does not yet flip the layer-3 top-k
boundary to Hugging Face token `229`. The next runtime slice should target the
QKV linear output precision path, where bf16 linear replay improved the
layer-0 K-norm outlier but did not fully close it.
