# 2026-06-01 Qwen Persistent Weight Materialization

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_weight_materialization.py`, which maps
  Qwen persistent weight descriptors through the `CudaPersistentDagTask`
  ctypes layout and binds concrete `tensor_args` pointers when a live pointer
  table is supplied.
- Wired the materialization plan into the Qwen serving scaffold, PTO serving
  preflight, CUDA examples manifest, example README, benchmark-viewer matrix,
  in-progress paper-readiness docs, dispatch log, and review-artifact tests.
- Captured current symbolic plan evidence at
  `tmp/cuda-backend/pto-serving-weight-materialization-b46497b3/qwen-persistent-weight-materialization.json`.

## Architecture Quality

The materializer reuses the same `CudaPersistentDagTask` ctypes structure used
by CUDA persistent DAG submission, so the review artifact reports the actual
host-side field offsets for `tensor_args`, `scalar_args`, and argument counts.
This keeps the document claim tied to executable code instead of a separate
JSON-only description.

The default artifact remains honest: without a decode-loop-owned live pointer
table, it emits symbolic `resident_weight_ptrs[slot_id]` sources rather than
pretending that freed probe allocations are usable runtime pointers.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_weight_materialization.py \
  --output-json \
  tmp/cuda-backend/pto-serving-weight-materialization-b46497b3/qwen-persistent-weight-materialization.json
```

Result: `status=persistent_weight_materialization_plan_ready`,
`materialized_task_count=255`, `symbolic_tensor_pointer_count=399`,
`bound_tensor_pointer_count=0`, and `missing_pointer_count=0`.

The focused TDD selector first failed because the materializer script did not
exist; after implementation the selected materialization and weight-argument
tests passed.

## Remaining Gaps

- Make the decode-loop runner own a live resident weight pointer table and
  call this materializer before persistent DAG submission.
- Bind runtime token IDs, allocate and bind KV-cache buffers, generate Qwen
  kernels, execute the decode loop, and import full-serving viewer rows for
  `Qwen/Qwen3-8B`.
