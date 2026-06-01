# CUDA Examples: Qwen Safetensors Shard Status

## Qwen Safetensors Shard Status

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_safetensors_fetch.py \
  --output-json tmp/cuda-backend/pto-serving-shards/qwen-safetensors-shards.json
```

Expected output: command exits 0; output JSON records Qwen safetensors shard
URLs, local target paths, present/missing counts, and resumable fetch commands
without downloading by default.

This is a placement and fetch-plan artifact, not a CUDA loader. Use
`--download` only when intentionally fetching the 16 GB Qwen3-8B safetensors
shards into `tmp/sources/qwen3-8b-safetensors/`; rerun the metadata probe after
all shards report `present`.

