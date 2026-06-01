# CUDA Examples: Qwen Weight Inventory

## Qwen Weight Inventory

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_weight_inventory.py \
  --output-json tmp/cuda-backend/pto-serving-weights/qwen-weight-inventory.json
```

Expected output: command exits 0; output JSON records safetensors shard count,
tensor count, binding groups, total size, the config-derived expected
shape/dtype contract, and the remaining tensor-open and CUDA weight-binding
gaps.

This is a weight inventory, not a weight loader. It parses the Qwen3-8B
safetensors index captured under `tmp/sources/` and makes the persistent-device
binding groups and expected tensor shapes reviewable before any model tensors
are opened or copied.

