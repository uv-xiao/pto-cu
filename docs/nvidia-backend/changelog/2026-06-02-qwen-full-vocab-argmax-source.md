# 2026-06-02 Qwen Full-Vocab Argmax Source

## Code And Data Changed

- Updated the generated Qwen logits shape path so device-side sampled-token
  feedback scans all descriptor `cols` instead of the previous four-token
  diagnostic subset.
- Added `qwen_logits_full_vocab_argmax_source` to the task-body manifest
  contracts.
- Kept the four-token loop only in the legacy diagnostic fallback path used by
  proxy tests without descriptor shapes.

## Architecture Quality

The logits task now ties linear logits and argmax feedback to the same
descriptor vocab width. This moves sampling from a fixed proxy subset toward
the model-shape source contract needed by PTO full-serving rows.

This is still source-level evidence. Paper-ready serving still requires full
attention correctness, complete token-by-token execution, and result import.

## Evaluation Run

- Passed:
  `.venv/bin/python -m pytest tests/ut/py/test_nvidia_qwen_task_body_math.py -q`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py --output-json tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-full-vocab-argmax.json --output-source tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-full-vocab-argmax.cu`.
- Passed:
  `/usr/local/cuda-12.8/bin/nvcc -ptx -arch=compute_80 tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-full-vocab-argmax.cu -o tmp/cuda-backend/pto-serving-task-bodies/qwen-persistent-task-bodies-full-vocab-argmax.ptx`.

## Remaining Gaps

PTO full-serving rows still require full attention, complete decode-loop
execution, and viewer import for both MPK-policy and VDCores-policy workloads.
