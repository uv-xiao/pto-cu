# 2026-06-04 Qwen QKV BF16 Boundary

## Code And Data Changed

- Updated `qwen_attention_qkv` so Q, K, and V projection outputs are rounded
  to the Hugging Face bf16 tensor boundary before feeding Q/K RMSNorm and KV
  cache writeback.
- Added one focused generated-source contract for the QKV boundary.
- Captured layer-0 and layer-3 MPK-policy full-row probes after the QKV
  boundary change.

## Architecture Quality

This is runtime numeric work, not a viewer or test-surface expansion. The
change sits at the model-equivalent Qwen projection boundary that feeds the
existing QK/RoPE drift path. The test remains a narrow source contract that
guards the dtype boundary rather than adding sparse row-specific assertions.

The new evidence confirms the QKV boundary improves the projection mismatch
at layer 0. It does not flip the layer-3 token boundary, so the next slice
should stay on QK/RoPE or downstream accumulation behavior rather than adding
more artifact-format checks.

## Evaluation Run

Focused source contract:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_qkv_projection_matches_hf_bf16_output_boundary -q
```

Result: passed (`1 passed`).

Task-body source suite:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: passed (`11 passed`).

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
`tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-qkv-bf16-boundary/`.

Result: the runner passed with zero scheduler errors. The full-row comparison
reports `layer_0_attention_qkv.mean_abs_delta=0.000030`,
`layer_0_attention_qk_norm.mean_abs_delta=0.001969`, and
`layer_0_mlp_down.mean_abs_delta=0.000199`.

Layer-3 post-fix probe:

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
`tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-qkv-bf16-boundary/`.

Result: the runner passed with zero scheduler errors. PTO first-512 top-k
remains `[10, 167, 376, 475, 58]`; Hugging Face first-512 top-k is
`[10, 167, 376, 475, 229]`. The layer-2 stage comparison reports
`layer_2_attention_qkv.mean_abs_delta=0.000218`,
`layer_2_attention_qk_norm.mean_abs_delta=0.004728`,
`layer_2_attention_o.mean_abs_delta=0.000453`,
`layer_2_post_attention_norm.mean_abs_delta=0.001830`, and
`layer_2_mlp_down.mean_abs_delta=0.000886`.

## Remaining Gaps

QKV output precision is no longer the leading layer-0 projection mismatch, but
full Qwen correctness remains open. The layer-3 boundary is still a close
token-rank mismatch between PTO token `58` and Hugging Face token `229`.
The next benchmark-model slice should continue at QK/RoPE precision and
downstream accumulation, not at viewer formatting or broad new test matrices.
