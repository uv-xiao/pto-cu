# CUDA Examples: Persistent Qwen Serving Scaffold

## Persistent Qwen Serving Scaffold

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_qwen_serving_scaffold.py \
  --output-json tmp/cuda-backend/pto-serving-scaffold/qwen-serving-scaffold.json
```

Expected output: command exits 0; output JSON records `status=partial` until
live Qwen tokenizer, weight loader, KV-cache lifecycle, task bodies,
decode-loop execution, and full-serving viewer import stages are complete.

This is not a benchmark result. It is the repo-owned lifecycle scaffold for
the PTO `Qwen/Qwen3-8B` full-serving work queue item.

