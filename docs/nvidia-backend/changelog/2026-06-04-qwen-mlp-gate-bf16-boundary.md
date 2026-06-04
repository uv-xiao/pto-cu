# 2026-06-04 Qwen MLP Gate BF16 Boundary

## Code And Data Changed

- Rounded generated `qwen_mlp_gate_up` writes to the Hugging Face bf16 tensor
  boundary after `silu(gate_proj) * up_proj`.
- Updated the existing generated-source contract instead of adding another
  artifact-specific test.
- Captured a fresh layer-3 MPK A100 probe with layer-2 MLP activation samples.

## Architecture Quality

This is a benchmark-model runtime slice. Qwen's MLP activation product is an
intermediate tensor consumed by `down_proj`; keeping it in FP32 preserved more
precision than the Hugging Face bf16 path and inflated the close token-58
ranking boundary. The generated task body now makes that tensor boundary
explicit.

The testing change stays compact. It converts the existing source contract
from "raw SiLU product" to "bf16 SiLU product" rather than introducing a new
row-by-row artifact matrix.

## Evaluation Run

Focused red/green source contract:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_generated_source_contains_qwen_unit_math_kernels \
  -q
```

Result: failed before the generated MLP task rounded the activation product;
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
`tmp/cuda-backend/qwen-prefill-layer3-mpk-1step-2026-06-04-mlp-gate-bf16-boundary/`.

Result: the runner passed with zero scheduler errors and diagnostic logits
reference `status=pass`, `max_abs_error=0.0`. PTO first-512 top-k remains
`[(10, 6.8125), (167, 5.96875), (376, 5.65625), (475, 5.53125),
(58, 5.46875)]`. This is not a full top-k match, but it moves PTO token `58`
from `5.5` to the Hugging Face selected-logit value `5.46875`.

## Remaining Gaps

Full Qwen correctness remains open. Hugging Face first-512 top-k remains
`[(10, 6.84375), (167, 6.03125), (376, 5.59375), (475, 5.53125),
(229, 5.5)]`; PTO still misses token `229` at rank five. The next slice should
continue upstream numeric root-cause work instead of changing top-k tie
ordering or adding viewer-only diagnostics.
