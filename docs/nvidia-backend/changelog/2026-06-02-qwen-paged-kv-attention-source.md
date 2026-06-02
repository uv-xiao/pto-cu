# 2026-06-02 Qwen Paged KV Attention Source

## Code And Data Changed

- Added `kv_page_table` as a runtime-generated tensor argument for
  `qwen_attention_o` descriptors.
- Added activation-workspace ownership for an identity logical-to-physical
  KV page table and included it in launch-packet runtime-buffer binding.
- Updated generated `qwen_attention_o` CUDA source to map logical decode
  steps through `tensor_args[1]` before reading mutable `c`/`d` KV cache.
- Updated paper-readiness viewer data so PTO full-serving remaining gates are
  tiled attention, full loop execution, and row import.

## Architecture Quality

The callable remains a single persistent-device task body. The page table is
bound through the same descriptor/runtime-buffer path already used for RoPE
tables, so the host runner owns lifecycle and pointer binding while the device
task body owns per-step cache address selection.

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
  --output-json tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-paged-kv.json \
  --output-source tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-paged-kv.cu

nvcc -ptx -arch=compute_80 \
  tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-paged-kv.cu \
  -o tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-paged-kv.ptx
```

The example, changelog, viewer-data, and NVIDIA review guards also passed.

## Remaining Gaps

PTO full-serving rows still require tiled attention promotion, full
decode-loop execution, and viewer result import for MPK and VDCores policy
rows.
