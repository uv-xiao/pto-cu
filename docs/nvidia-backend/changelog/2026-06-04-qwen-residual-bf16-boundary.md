# 2026-06-04 Qwen Residual BF16 Boundary

## Code And Data Changed

- Rounded model-equivalent Qwen attention-O output, post-attention residual
  input, and MLP residual output to Hugging Face bf16 tensor boundaries.
- Added one focused generated-source contract for these residual-stream
  boundaries.
- Captured a layer-0 full-row MPK probe and a layer-3 slim full-row probe for
  the main upstream rows after the residual-stream boundary change.

## Architecture Quality

This keeps the work on benchmark-model runtime correctness. Hugging Face
returns bf16 tensors from attention output projection and decoder-layer
residual additions, while the generated Qwen path was carrying those residual
stream values as FP32. The source change narrows that semantic gap without
adding a new broad test matrix.

The generated-source contract is intentionally focused on boundary placement:
attention-O output, the residual value normalized by post-attention RMSNorm,
and the MLP down residual output.

## Evaluation Run

Focused source contract:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_qwen_residual_stream_matches_hf_bf16_boundaries -q
```

Result: passed (`1 passed`).

Task-body source suite:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: passed (`13 passed`).

Layer-0 post-fix probe:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  timeout 420s .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-prefill-prompt \
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
`tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-residual-bf16-boundary/`.

Result: the runner passed with zero scheduler errors. Compared with the
RMSNorm boundary artifact, layer-0 full-row means improve for
`layer_0_attention_o` (`0.000099` to `0.000089`),
`layer_0_post_attention_norm` (`0.000162` to `0.000158`), and
`layer_0_mlp_down` (`0.000216` to `0.000204`). The MLP-down p99 rises to a
bf16 quantum, so this is a semantic boundary correction rather than a complete
numeric fix.

Layer-3 slim post-fix probe:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  timeout 420s .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-prefill-prompt \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-decode-steps 1 \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 3 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-logits-active-cols 512 \
  --resource-backed-projection-active-cols full \
  --resource-backed-numeric-task-mode model_equivalent \
  --resource-backed-activation-row-dump-descriptor-ids \
  layer_2_input_norm,layer_2_attention_qkv,layer_2_attention_qk_norm
```

Artifact:
`tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-residual-bf16-boundary-slim/`.

Result: the runner completed within the timeout and passed with zero scheduler
errors. PTO first-512 top-k remains `[10, 167, 376, 475, 58]`; Hugging Face
remains `[10, 167, 376, 475, 229]`. Compared with the RMSNorm boundary slim
artifact, the layer-2 stage means improve for `layer_2_attention_qkv`
(`0.000204` to `0.000194`) and `layer_2_attention_qk_norm` (`0.004423` to
`0.004318`), with QK/RoPE p99 improving from `0.03125` to `0.023438`.

## Remaining Gaps

Full Qwen correctness remains open. The residual-stream boundaries reduce
distributed drift but do not flip the close layer-3 first-512 top-k boundary
between PTO token `58` and Hugging Face token `229`.
