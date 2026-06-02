# Qwen Attention-O Projection Window

## Code And Data Changed

- Extended `--resource-backed-projection-active-cols` to cover
  `qwen_attention_o` projection inputs.
- Kept `qwen_attention_o.task_shape_fields.scalar1` as the attention tile size
  and added a separate descriptor field,
  `attention_o_projection_input_count`, for `scalar_args[1]`.
- Updated resource-backed graph-materialization tests to assert descriptor
  rewriting and launch-packet scalar argument binding.

## Architecture Quality

This removes the last hidden projection-window cap from first-layer Qwen
resource-backed runs. QKV, attention-output, MLP gate/up, and MLP down
projection windows are now controlled by one runner option while preserving
the existing attention tile-size field.

## Evaluation Run

Focused Python verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q
```

Result: `37 passed`.

A100 first-layer smoke with a bounded `256` projection window passed:

```text
tmp/cuda-backend/qwen-attention-o-projection-window-256-first-layer-2026-06-03/
```

The artifact records `qwen_attention_o` with
`attention_o_projection_input_count=256`, completed 10 tasks with zero
scheduler errors, and passed the diagnostic logits projection reference.

## Remaining Gaps

A full first-layer projection run with `--resource-backed-projection-active-cols
full` is impractical with the current naive `qwen_attention_o` body because it
nests output projection, KV-window, and head-dim loops per output element.
Full-serving correctness still needs a tiled or reduced attention-output
implementation before full projection windows can be used in paper-scale runs.
