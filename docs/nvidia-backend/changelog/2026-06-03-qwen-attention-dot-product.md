# Qwen Attention Dot Product

## Code And Data Changed

- Updated the generated `qwen_attention_o` CUDA task body to compute decode
  attention scores as a full query-head dot product against each cached key,
  scaled by `sqrt(head_dim)`.
- Updated the Qwen decode-attention oracle from per-channel scores to the same
  head-level dot-product equation.
- Added the manifest contract
  `qwen_decode_attention_dot_product_source`.

## Architecture Quality

This moves `qwen_attention_o` closer to the model-correct Qwen decode path.
The old diagnostic body computed softmax scores independently per hidden
channel. The new body computes one score per query head and cache token, then
uses the resulting weights to gather the value component for each output
channel before applying the O projection.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_qwen_single_context_session.py -q
```

Result: `63 passed`.

A100 first-layer live-session smoke passed:

```text
tmp/cuda-backend/qwen-attention-dot-product-first-layer-2026-06-03/
```

The artifact reports 18 prompt-prefill positions, 144 completed prefill tasks,
a 2-task readout-only first decode, a 10-task full selected-DAG second decode,
device token feedback for both decode steps, and zero scheduler errors.

## Remaining Gaps

This is still diagnostic Qwen evidence. Full Qwen numerical correctness against
the Hugging Face model and policy-length MPK/VDCores serving captures remain
open.
