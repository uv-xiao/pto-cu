# 2026-06-02 Qwen Logits Projection Reference

## Code And Data Changed

- Replaced the resource-backed logits diagnostic reference with a bounded
  hidden-by-vocab projection reference matching the generated logits task body.
- Removed the obsolete toy diagnostic formula helper from
  `qwen_decode_loop_runner_impl/resource_graph.py`.
- Updated future resource-backed viewer imports to use the
  `diagnostic_qwen_tiled_vocab_projection` symbol.
- Added inspectable evidence under
  `tmp/cuda-backend/pto-serving-logits-projection-reference-2026-06-02/`.

## Architecture Quality

The resource-backed diagnostic path now checks the same row/column projection
shape as the CUDA source: `row = i / cols`, `col = i % cols`, then a hidden
loop over descriptor-controlled strides. Large full-vocab references are capped
and reported as `not_checked` instead of being promoted with an unrelated
formula.

## Evaluation Run

Focused verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py -q
```

Result: `13 passed in 0.10s`.

## Remaining Gaps

This is still diagnostic reference alignment. Full Qwen numerical correctness
and full-serving row import remain open.
