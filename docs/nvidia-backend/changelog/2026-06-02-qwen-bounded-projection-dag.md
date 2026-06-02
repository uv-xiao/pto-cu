# Qwen Bounded Projection DAG

## Code And Data Changed

- Added an explicit diagnostic active-column bound for heavy Qwen projection
  tasks: `qwen_attention_qkv`, `qwen_mlp_gate_up`, and `qwen_mlp_down`.
- The generated persistent-device task bodies now skip expensive matrix inner
  loops outside `active_projection_cols` and zero those inactive output
  positions.
- Imported compact benchmark-viewer rows for full 255-task MPK-policy and
  VDCores-policy diagnostic DAG runs. The raw JSON artifacts remain under
  `tmp/cuda-backend/qwen-projection-bounded/`.

## Architecture Quality

This keeps the diagnostic bound explicit in task descriptors instead of hiding
it in the runner. The same descriptor materialization, launch packet builder,
persistent-device scheduler, resident weight table, activation workspace, and
logits reference checker are used by the full resource-backed path.

The implementation does not claim paper-grade full-serving correctness. It
turns the current scalar diagnostic Qwen graph into an executable full-DAG
runtime target so the remaining tensor-core or tiled-projection work can be
measured against a working end-to-end persistent-device baseline.

## Evaluation Run

Generated source evidence was written under
`tmp/cuda-backend/qwen-projection-bounded/`.

Full 255-task diagnostic DAG live runs passed on A100:

| Workload | Status | Completed | Device time | Logits reference |
| -------- | ------ | --------- | ----------- | ---------------- |
| `mpk_offline_decode` | pass | 255 | 42871.976 ms | pass |
| `vdcores_offline_decode` | pass | 255 | 43109.994 ms | pass |

Both runs reported `partial_logits_not_full_vocab` with
`diagnostic_qwen_tiled_vocab_projection` reference checks passing on the
bounded logits window.

Focused source and descriptor tests passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py -q -k \
  generated_source_contains_qwen_unit_math_kernels

PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py -q -k \
  qwen_weight_descriptors_emit_callable_shape_fields
```

## Remaining Gaps

The committed viewer rows are diagnostic full-DAG evidence, not paper-ready
full-serving evidence. Full Qwen numerical correctness still requires
unbounded vocabulary logits and production-grade tiled or tensor-core
projection kernels rather than scalar one-block projection loops with
diagnostic active-column limits.
