# 2026-06-04 Qwen RMSNorm BF16 Boundary

## Code And Data Changed

- Rounded model-equivalent Qwen input RMSNorm, post-attention RMSNorm, and
  final RMSNorm outputs to the Hugging Face bf16 tensor boundary.
- Added one focused generated-source contract for the RMSNorm output
  boundaries.
- Captured a layer-0 full-row MPK probe and a layer-3 slim full-row probe for
  the main upstream rows after the boundary change.

## Architecture Quality

This is benchmark-model runtime work, not a viewer or broad test expansion.
The source change follows the local Hugging Face `Qwen3RMSNorm` contract:
RMSNorm computes in float32, casts the normalized activation back to the input
dtype, then returns a bf16 tensor in the loaded local reference.

The test is intentionally narrow. It guards the generated source boundary once
instead of adding sparse row-by-row artifact assertions.

## Evaluation Run

Focused source contract:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_qwen_rmsnorm_outputs_match_hf_bf16_boundary -q
```

Result: passed (`1 passed`).

Task-body source suite:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: passed (`12 passed`).

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
`tmp/cuda-backend/qwen-prefill-layer1-mpk-1step-2026-06-04-rmsnorm-bf16-boundary/`.

Result: the runner passed with zero scheduler errors. Compared with the QKV
boundary artifact, layer-0 full-row means improve for
`layer_0_input_norm` (`0.000007` to `0.000005`),
`layer_0_attention_qkv` (`0.000030` to `0.000025`), and
`layer_0_attention_qk_norm` (`0.001969` to `0.001333`). The layer-0
`attention_qk_norm.max_abs_delta` drops from `1.0` to `0.0625`.

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
`tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-rmsnorm-bf16-boundary-slim/`.

Result: the runner completed within the timeout and passed with zero scheduler
errors. PTO first-512 top-k remains `[10, 167, 376, 475, 58]`; Hugging Face
remains `[10, 167, 376, 475, 229]`. Compared with the QKV boundary artifact,
the slim layer-2 stage means improve for `layer_2_input_norm` (`0.000087` to
`0.000082`), `layer_2_attention_qkv` (`0.000218` to `0.000204`), and
`layer_2_attention_qk_norm` (`0.004728` to `0.004423`).

## Remaining Gaps

Full Qwen correctness remains open. RMSNorm output boundaries reduce upstream
drift but do not flip the layer-3 first-512 top-k boundary. The next slice
should continue investigating accumulated layer-state drift and the close
token `58` versus token `229` rank boundary.
