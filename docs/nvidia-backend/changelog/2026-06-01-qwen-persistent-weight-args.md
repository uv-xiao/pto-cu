# 2026-06-01 Qwen Persistent Weight Arguments

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_weight_args.py`, which converts Qwen
  CUDA weight binding slots into persistent DAG task argument descriptors.
- Wired the manifest into the Qwen serving scaffold, PTO serving preflight,
  CUDA examples manifest, example README, benchmark-viewer matrix,
  in-progress paper-readiness docs, and review-artifact tests.
- Captured real Qwen/Qwen3-8B evidence at
  `tmp/cuda-backend/pto-serving-weight-args-21589e81/qwen-persistent-weight-args.json`.

## Architecture Quality

The branch now makes the current persistent DAG ABI constraint explicit:
`PtoCudaPersistentDagTask` exposes four generic `tensor_args` pointers per
task. Qwen layer work is decomposed into smaller descriptors such as attention
QKV, attention Q/K norm, MLP gate/up, and MLP down so every descriptor fits
that ABI while covering all validated weights.

This still does not claim decode execution. It is a reviewable contract for
runtime task-descriptor materialization and generated kernel bodies.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_weight_args.py \
  --weight-binding-json \
  tmp/cuda-backend/pto-serving-weight-residency-1ae913c9/qwen-cuda-weight-residency.json \
  --output-json \
  tmp/cuda-backend/pto-serving-weight-args-21589e81/qwen-persistent-weight-args.json
```

Result: `status=persistent_weight_args_ready`,
`task_arg_descriptor_count=255`, `covered_tensor_count=399`,
`missing_tensor_count=0`, `uncovered_binding_count=0`, and
`max_tensor_args_per_task=3`.

## Remaining Gaps

- Materialize resident device pointers into persistent DAG task descriptors at
  runtime.
- Generate Qwen kernels that consume these descriptors.
- Bind runtime token IDs, allocate and bind KV-cache buffers, execute the
  decode loop, and import full-serving viewer rows for `Qwen/Qwen3-8B`.
