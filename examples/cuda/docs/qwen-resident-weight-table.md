# CUDA Examples: Qwen Resident Weight Table

## Qwen Resident Weight Table

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_resident_weight_table.py \
  --output-json tmp/cuda-backend/pto-serving-resident-weight-table/qwen-resident-weight-table.json
```

Expected output: command exits 0; output JSON records the process-scoped
resident weight pointer owner lifecycle, materialization bridge, pointer
count, and teardown count; add `--cuda-live` to allocate and copy through the
CUDA runtime.

The default mode is `dry_run_pointer_lifecycle`: it exercises the same owner,
pointer-table, materialization, and close/free ordering without allocating
16.38 GB again. Use `--cuda-live` only inside a runner process that will submit
the persistent DAG while the owner is still open.
