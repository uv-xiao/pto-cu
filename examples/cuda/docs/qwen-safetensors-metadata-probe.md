# CUDA Examples: Qwen Safetensors Metadata Probe

## Qwen Safetensors Metadata Probe

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_safetensors_metadata.py \
  --weight-inventory-json tmp/cuda-backend/pto-serving-weights/qwen-weight-inventory.json \
  --output-json tmp/cuda-backend/pto-serving-safetensors/qwen-safetensors-metadata.json
```

Expected output: command exits 0; output JSON records whether Qwen
safetensors shard headers were opened and whether actual tensor shapes/dtypes
match the expected contract.

This is a metadata probe, not a CUDA loader. When the Qwen shards are absent it
reports `shards_missing`; when shards are present it parses standard
safetensors headers and validates tensor shape/dtype metadata before any data
copy or CUDA binding step.

