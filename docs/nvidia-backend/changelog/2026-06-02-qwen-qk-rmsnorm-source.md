# 2026-06-02 Qwen QK RMSNorm Source

## Code And Data Changed

- Updated `qwen_attention_qk_norm` generated CUDA source to use descriptor
  `cols`, `inner`, and `lda` fields for row-wise RMS scale computation.
- The task now reads Q and K norm weights through persistent DAG
  `tensor_args[0]` and `tensor_args[1]`, then applies the averaged weight
  scale to the normalized row value.
- Added `qwen_shape_field_qk_rmsnorm_source` as an explicit task-body evidence
  contract and tightened the unit test to check the rendered CUDA source.

## Architecture Quality

This narrows the attention source gap without changing the persistent DAG ABI.
The generated task body keeps the old diagnostic fallback for descriptors that
do not carry shape metadata, while the Qwen descriptor path now has readable
shape-field source evidence for the QK norm stage.

## Evaluation Run

- Passed:
  `.venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py --output-json tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rmsnorm.json --output-source tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rmsnorm.cu`.
- Passed:
  `/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rmsnorm.cu -o tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-qk-rmsnorm.ptx`.

## Remaining Gaps

PTO full-serving rows still require RoPE, decode attention reduction,
complete decode-loop execution, and viewer import for both MPK-policy and
VDCores-policy workloads.
