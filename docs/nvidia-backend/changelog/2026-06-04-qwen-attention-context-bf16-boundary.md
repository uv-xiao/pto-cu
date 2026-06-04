# 2026-06-04 Qwen Attention Context BF16 Boundary

## Code And Data Changed

- Rounded the softmax-weighted attention value vector to the Hugging Face bf16
  tensor boundary before output projection.
- Updated the existing generated-source contract for `qwen_attention_o`.
- Captured a fresh layer-3 MPK A100 probe with the same first-512 logits and
  layer-2 stage samples used by the prior boundary slices.

## Architecture Quality

This keeps the work on benchmark-model runtime correctness. In the Hugging
Face bf16 path, the attention context tensor is consumed by `o_proj` as a bf16
activation. The generated Qwen task was preserving that context as FP32 before
the output projection, which can skew close rank boundaries.

The test remains a compact source contract. It replaces the previous
"attention context is assigned" assertion with a boundary assertion, rather
than adding another artifact-specific row test.

## Evaluation Run

Focused red/green source contract:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
  -q
```

Result: failed before the generated attention task rounded the context vector;
passed after the runtime change (`1 passed`).

Task-body source suite:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q
```

Result: passed (`14 passed`).

Runtime probe:

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
  layer_2_input_norm,layer_2_attention_qkv,layer_2_attention_qk_norm,layer_2_attention_o,layer_2_post_attention_norm,layer_2_mlp_gate_up,layer_2_mlp_down
```

Artifact:
`tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-attention-context-bf16-boundary/`.

Result: the runner passed with zero scheduler errors and diagnostic logits
reference `status=pass`, `max_abs_error=0.0`. PTO first-512 top-k remains
`[(10, 6.84375), (167, 6.0), (376, 5.625), (475, 5.53125), (58, 5.5)]`.
This improves the common-token logit error for tokens `10`, `167`, and `376`
relative to the MLP-gate boundary artifact, but it does not close the top-k
match.

## Remaining Gaps

Full Qwen correctness remains open. Hugging Face first-512 top-k remains
`[(10, 6.84375), (167, 6.03125), (376, 5.59375), (475, 5.53125),
(229, 5.5)]`; PTO still misses token `229` at rank five and token `58` moved
back to `5.5`. The next slice should continue upstream numeric root-cause work
instead of changing top-k tie ordering or adding viewer-only diagnostics.
