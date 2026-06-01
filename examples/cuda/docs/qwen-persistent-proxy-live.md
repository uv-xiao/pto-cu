# CUDA Examples: Qwen Persistent Proxy Live

## Qwen Persistent Proxy Live

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_proxy_live.py \
  --device 0 \
  --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-proxy-live/qwen-proxy-live.json
```

Expected output: command exits 0 on a CUDA host; output JSON records
status=pass for a controlled single-task Qwen QKV proxy launched through
cuda/persistent_device.

This is live runtime evidence for the controlled proxy only. It proves the
generated QKV task body can be compiled, prepared, launched by the
persistent-device scheduler, and copied back with mutable `c`/`d` KV fields.
It is not a full Qwen decode loop or a numerically correct Qwen kernel.

