# CUDA Examples: Qwen Persistent Weight Materialization

## Qwen Persistent Weight Materialization

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_weight_materialization.py \
  --output-json tmp/cuda-backend/pto-serving-weight-materialization/qwen-persistent-weight-materialization.json
```

Expected output: command exits 0; output JSON records how Qwen persistent
weight task descriptors are materialized through the `CudaPersistentDagTask`
ctypes layout, and binds resident device pointers when a live pointer table is
supplied.

Without `--pointer-table-json`, this emits a symbolic materialization plan that
uses `resident_weight_ptrs[slot_id]` as the source for each `tensor_args`
entry. With a live pointer table from the decode-loop runner, it emits concrete
device addresses and validates that each pointer matches the expected tensor
slot before DAG submission.

