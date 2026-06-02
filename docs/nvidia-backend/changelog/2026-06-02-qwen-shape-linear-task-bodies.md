# 2026-06-02 Qwen Shape-Linear Task Bodies

## Code And Data Changed

- Added a generated CUDA helper for descriptor-shape linear projections over
  `rows`, `cols`, `inner`, `lda`, and `ldb`.
- Updated Qwen QKV, attention output, MLP gate/up, MLP down, and logits task
  bodies to use shape-field matrix reductions when tensor pointers and shape
  metadata are present.
- Kept the old sampled diagnostic formulas as fallback paths for small proxy
  tests without full descriptor shapes.

## Architecture Quality

The generated persistent-device source now consumes the same callable-local
matrix-shape ABI carried by Qwen resident descriptors and launch packets. This
connects descriptor planning, workspace sizing, and task-body math through one
reviewable contract instead of separate proxy formulas.

The change does not claim full Qwen serving correctness. It removes the
projection/logits source-shape blocker while leaving RoPE, paged attention,
full attention reduction, and sampling as explicit runtime gaps.

## Evaluation Run

- Passed:
  `.venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q`.
- Passed:
  `.venv/bin/python -m pytest tests/ut/py/test_cuda_persistent_codegen.py -q -k 'persistent_device'`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py --output-json tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-shape-linear.json --output-source tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-shape-linear.cu`.
- Passed:
  `/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-shape-linear.cu -o tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-shape-linear.ptx`.

## Remaining Gaps

PTO full-serving rows still require numerically complete Qwen attention,
sampling, token-by-token decode-loop execution, and viewer import for both
MPK-policy and VDCores-policy workloads.
