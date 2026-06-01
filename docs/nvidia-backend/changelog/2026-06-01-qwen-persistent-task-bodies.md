# 2026-06-01 Qwen Persistent Task Bodies

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_task_bodies.py` plus
  `examples/cuda/qwen_persistent_task_bodies_impl/`.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-task-bodies-2026-06-01/`
  `qwen-persistent-task-bodies.json` and
  `qwen-persistent-task-bodies.cu`.

## Architecture Quality

The artifact uses the existing persistent DAG source generator instead of a
separate source shape. It reports token field consumption through `a`, `b`,
and `out`, KV-cache field consumption through mutable `c` and `d`, and weight
consumption through `tensor_args`.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_task_bodies.py \
  --output-json \
  tmp/cuda-backend/pto-serving-task-bodies-2026-06-01/\
qwen-persistent-task-bodies.json \
  --output-source \
  tmp/cuda-backend/pto-serving-task-bodies-2026-06-01/\
qwen-persistent-task-bodies.cu
```

Result: `status=generated_task_bodies_ready`, 10 rendered task bodies, and
source sha256
`92344f30355981ac7777320e8670377df614cff46b62917b70395f7171c90b4f`.

## Remaining Gaps

- Replace source-level task bodies with numerically correct Qwen kernels.
- Execute the `cuda_live` decode loop and import full-serving viewer rows.
