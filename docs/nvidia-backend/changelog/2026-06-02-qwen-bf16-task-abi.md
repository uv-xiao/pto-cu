# 2026-06-02 Qwen BF16 Task ABI

## Code And Data Changed

- Added `tensor_arg_dtypes[4]` to `CudaPersistentDagTask` and the matching
  `PtoCudaPersistentDagTask` C header.
- Added a generated CUDA helper that reads float32 tensor arguments directly
  and converts BF16 tensor arguments with the repo's `DataType::BFLOAT16`
  code `6`.
- Packed Qwen descriptor `tensor_arg_metadata[].dtype` into launch packets.
- Updated generated Qwen task bodies to read weights through the dtype-aware
  helper while preserving float32 diagnostic behavior by default.

## Architecture Quality

The persistent-device task ABI now carries the missing information needed to
consume real Qwen3-8B resident weights. Previous descriptor metadata recorded
that Qwen weights are BF16, but device task bodies only saw pointer slots. The
new dtype slots keep pointer ABI compatibility and let task bodies branch on
the same dtype code used by the repo-wide `DataType` enum.

This narrowed the full-serving gap from "device code cannot know resident
weight dtype" to kernel-math work. Follow-up source work now covers
shape-field projection, logits linear bodies, and full-vocab argmax source;
attention and full decode-loop execution remain open.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_cuda_persistent_codegen.py tests/ut/py/test_nvidia_qwen_graph_materialization.py tests/ut/py/test_nvidia_qwen_task_body_math.py`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py --output-json tmp/cuda-backend/qwen-bf16-dtype-abi/task-bodies.json --output-source tmp/cuda-backend/qwen-bf16-dtype-abi/task-bodies.cu`.
- Passed:
  `/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 tmp/cuda-backend/qwen-bf16-dtype-abi/task-bodies.cu -o tmp/cuda-backend/qwen-bf16-dtype-abi/task-bodies.ptx`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_review_artifacts.py::test_qwen_persistent_task_bodies_render_generated_source`.

## Remaining Gaps

The task bodies are now dtype-aware, but full-serving rows still need RoPE/QK
norm, paged attention, full attention reduction, and complete decode-loop
execution.
