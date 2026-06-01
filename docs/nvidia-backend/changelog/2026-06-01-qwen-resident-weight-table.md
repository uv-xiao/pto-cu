# 2026-06-01 Qwen Resident Weight Table

## Code And Data Changed

- Added `examples/cuda/qwen_resident_weight_table.py`, which owns a
  process-scoped Qwen resident weight pointer table, exposes it while open,
  feeds it to persistent weight materialization, and frees pointers on close.
- Extended `qwen_persistent_weight_materialization.py` so it can consume an
  in-memory pointer table, not only a JSON file written before materialization.
- Wired the resident-table lifecycle into the Qwen serving scaffold, PTO
  serving preflight, CUDA examples manifest, example README,
  benchmark-viewer matrix, in-progress paper-readiness docs, dispatch log, and
  review-artifact tests.
- Captured current dry-run lifecycle evidence at
  `tmp/cuda-backend/pto-serving-resident-weight-table-2026-06-01/qwen-resident-weight-table.json`.

## Architecture Quality

The new owner makes pointer lifetime explicit. A pointer table is marked
`valid_until_owner_close` while the owner is open, and `closed` after teardown.
This avoids treating stale pointer JSON as a durable CUDA allocation.

The default artifact uses `dry_run_pointer_lifecycle` because the full CUDA
residency probe already proved 16.38 GB of device residency separately. The
same owner has a `--cuda-live` path for the decode-loop runner, where the
persistent DAG must be submitted while the owner remains open.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_resident_weight_table.py \
  --output-json \
  tmp/cuda-backend/pto-serving-resident-weight-table-2026-06-01/qwen-resident-weight-table.json
```

Result: `status=resident_weight_table_lifecycle_ready`,
`mode=dry_run_pointer_lifecycle`, `pointer_count=399`,
`bound_tensor_pointer_count=399`, and `freed_pointer_count=399`.

The focused TDD selector first failed because the resident weight table script
did not exist; after implementation the selected resident-table, preflight,
and scaffold tests passed.

## Remaining Gaps

- Run this owner in `cuda_live` mode from the decode-loop runner and keep it
  open through persistent DAG submission.
- Bind runtime token IDs, allocate and bind KV-cache buffers, generate Qwen
  kernels, execute the decode loop, and import full-serving viewer rows for
  `Qwen/Qwen3-8B`.
