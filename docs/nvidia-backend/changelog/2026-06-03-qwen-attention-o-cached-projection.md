# Qwen Attention-O Cached Projection

## Code And Data Changed

- Converted generated `qwen_attention_o` from element-threaded projection to a
  block-threaded task body.
- Added a shared `attention_values[4096]` cache so each bounded attention
  value is computed once per batch row, then reused across output columns.
- Added `qwen_attention_o_cached_projection_source` as explicit source
  evidence in the task-body manifest and example symbol list.

## Architecture Quality

This removes the full-projection blowup from the first-layer Qwen diagnostic
path. The old implementation recomputed the KV-window softmax for every output
column and projection column. The new task computes attention values once,
synchronizes the worker block, and then applies `o_proj_weight` over the cached
values.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py -q
```

Result: `50 passed`.

A100 first-layer smoke with full projection windows passed:

```text
tmp/cuda-backend/qwen-attention-o-cached-full-projection-first-layer-2026-06-03/
```

The artifact records full projection policy values for QKV, attention-output,
MLP gate/up, and MLP down. It completed 10 tasks with zero scheduler errors,
passed the diagnostic logits projection reference, and reported device time
`6490235904` ns.

## Remaining Gaps

This is still a first-layer diagnostic run with the default bounded logits
window. Full-serving promotion still requires all selected layers, full logits,
policy-length MPK and VDCores runs, and Hugging Face token/logit agreement.
