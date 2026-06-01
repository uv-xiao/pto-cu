# CUDA Examples: Qwen Persistent Decode Arguments

## Qwen Persistent Decode Arguments

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_decode_args.py \
  --output-json tmp/cuda-backend/pto-serving-decode-args/qwen-persistent-decode-args.json
```

Expected output: command exits 0; output JSON records how Qwen token device
pointers bind to persistent DAG a/b/out fields while preserving tensor_args
for weights.

This is a persistent decode task-argument artifact. Without a live token
pointer table from the decode-loop runner it records symbolic pointer sources;
with a token pointer table it validates concrete `a`, `b`, and `out` bindings.

