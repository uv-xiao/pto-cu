# CUDA Examples: Qwen KV-Cache Binding

## Qwen KV-Cache Binding

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_kv_cache_binding.py \
  --cuda-live \
  --device 0 \
  --output-json tmp/cuda-backend/pto-serving-kv-cache-live/qwen-kv-cache-binding.json
```

Expected output: command exits 0; output JSON records CUDA-live key/value
KV-cache allocation and pointer binding evidence.

The artifact derives KV-cache sizes from the Qwen serving lifecycle plan,
splits each planned cache into key and value buffers, allocates the planned
cache on CUDA, and maps them to the persistent DAG `c` and `d` fields. This
does not prefill KV values or prove Qwen attention correctness.

