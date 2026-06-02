# 2026-06-02 Qwen Weight Descriptor Shapes

## Code And Data Changed

- Added a local Qwen task-shape contract for persistent weight descriptors.
- Generated per-callable `task_shape_fields` for embedding, norm, QKV,
  attention output, MLP, and logits descriptors.
- Exposed the descriptor shape contract in the weight-argument manifest.

## Architecture Quality

Weight descriptors now carry callable-local matrix dimensions before they reach
resident-weight materialization or launch-packet packing. Workload rows still
come from decode args, while descriptor fields describe model dimensions such
as hidden width, fused QKV width, MLP intermediate width, and vocabulary size.

This narrows the next kernel implementation step: projection kernels can read
`rows`, `cols`, `inner`, `lda`, `ldb`, and `ldc` from the CUDA task ABI instead
of relying on hardcoded Qwen3-8B constants inside device code.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_graph_materialization.py`.

## Remaining Gaps

The descriptors now expose model-shape metadata, but the persistent task bodies
still need numerically complete Qwen projection, attention, MLP, normalization,
logits, and sampling implementations before full-serving rows can be imported.
