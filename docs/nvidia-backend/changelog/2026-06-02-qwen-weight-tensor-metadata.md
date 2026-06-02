# 2026-06-02 Qwen Weight Tensor Metadata

## Code And Data Changed

- Preserved weight tensor `dtype`, `shape`, and `size_bytes` as descriptor
  sidecar metadata.
- Preserved that sidecar through resident-weight materialization.
- Added an explicit `qwen_weight_tensor_metadata_contract` manifest contract.

## Architecture Quality

The persistent-device path now keeps Qwen weight metadata adjacent to pointer
arguments without changing the four-pointer CUDA task ABI. This matters because
Qwen3-8B resident weights are BF16 while the current generated task ABI exposes
weight slots as pointer fields. Future BF16-aware task bodies can now be
reviewed against descriptor metadata instead of hidden assumptions.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_graph_materialization.py`.

## Remaining Gaps

Follow-up work added the BF16 read path and shape-field projection/logits
linear source. PTO full-serving rows still require full attention, sampling,
and complete decode-loop execution before they can be imported.
