# 2026-06-02 Qwen QK RoPE Source

## Code And Data Changed

- Extended `qwen_attention_qk_norm` generated CUDA source with pairwise RoPE
  rotation over descriptor head-dim rows.
- The task reads optional cos/sin tables through persistent DAG
  `tensor_args[2]` and `tensor_args[3]`, with scalar identity fallback so
  existing descriptors still compile.
- Added `qwen_shape_field_qk_rope_source` as an explicit task-body evidence
  contract and updated review docs, CUDA example metadata, and paper-readiness
  data.

## Architecture Quality

This keeps QK RMSNorm and RoPE in the same generated task body, matching the
current descriptor granularity for `qwen_attention_qk_norm`. It avoids a new
ABI field while making the missing RoPE table binding explicit for the next
runtime wiring slice.

## Evaluation Run

- Passed:
  `.venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py --output-json tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rope.json --output-source tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rope.cu`.
- Passed:
  `/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rope.cu -o tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rope.ptx`.

## Remaining Gaps

PTO full-serving rows still require model RoPE table binding, decode attention
reduction, complete decode-loop execution, and viewer import for both
MPK-policy and VDCores-policy workloads.
