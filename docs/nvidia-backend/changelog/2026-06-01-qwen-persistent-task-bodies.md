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
and `out`, KV-cache field consumption through `c` and `d`, and weight
consumption through `tensor_args`. It also records that the current `c`/`d`
ABI is `const float *`, so mutable KV-cache writeback remains unresolved.

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
`89a292d05f63fa5a70442428f8329da114050e5bad8f3ee420bae6757c8f8875`.

## Remaining Gaps

- Replace source-level task bodies with numerically correct Qwen kernels.
- Add mutable KV-cache writeback support for the persistent DAG `c`/`d` ABI.
- Execute the `cuda_live` decode loop and import full-serving viewer rows.
