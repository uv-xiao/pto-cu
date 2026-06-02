# 2026-06-02 Qwen Logits Tile Descriptor

## Code And Data Changed

- Added `LOGITS_TILE_SIZE = 256` to the Qwen task-shape contract and emitted it
  as `qwen_logits.task_shape_fields.scalar0`.
- Added a focused graph-materialization assertion proving the descriptor-level
  logits tile-size field is present.
- Added review evidence under
  `tmp/cuda-backend/pto-serving-logits-tile-descriptor-2026-06-02/`.

## Architecture Quality

The launch descriptor now owns the logits tile-size value consumed by the
generated CUDA logits projection source. That removes the implicit fallback-only
contract between weight descriptors, launch packets, and task bodies.

## Evaluation Run

Focused verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_qwen_weight_descriptors_emit_callable_shape_fields -q
```

Result: `1 passed`.

## Remaining Gaps

This is descriptor/source contract alignment. Full Qwen numerical correctness
and full-serving row import remain open.
