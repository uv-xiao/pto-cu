# 2026-06-02 Qwen Tiled Attention Source

## Code And Data Changed

- Added descriptor `scalar1` as the Qwen attention tile-size field for
  `qwen_attention_o`.
- Updated generated `qwen_attention_o` CUDA source so both softmax passes
  iterate over bounded `attention_tile` windows before reading paged KV-cache
  entries.
- Added `qwen_tiled_decode_attention_softmax_source` to the task-body
  evidence symbols.
- Updated paper-readiness viewer data so PTO full-serving remaining gates are
  full loop execution and row import.

## Architecture Quality

The tile size stays in the same task-shape contract as page size and attention
head layout, so the persistent-device callable remains one source path. The
host-side descriptor controls the tile policy; device code uses it without
introducing a separate attention kernel ABI.

## Evaluation Run

Verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  tests/ut/py/test_nvidia_review_artifacts.py -q
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py \
  --output-json tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-tiled-attention.json \
  --output-source tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-tiled-attention.cu

nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-tiled-attention.cu \
  -o tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-tiled-attention.ptx
```

The example, changelog, viewer-data, and NVIDIA review guards also passed.

## Remaining Gaps

PTO full-serving rows still require end-to-end resource-backed decode-loop
execution and viewer result import for MPK and VDCores policy rows.
