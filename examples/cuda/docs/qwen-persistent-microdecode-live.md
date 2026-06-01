# CUDA Examples: Qwen Persistent Microdecode Live

## Qwen Persistent Microdecode Live

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_microdecode_live.py \
  --device 0 \
  --arch compute_80 \
  --repeat-runs 3 \
  --output-json tmp/cuda-backend/pto-serving-decode-loop-live/qwen-microdecode-loop.json
```

Expected output: command exits 0 on a CUDA host; output JSON records
status=pass for a controlled Qwen QKV-to-logits proxy DAG submitted
repeatedly through cuda/persistent_device.

This is the smallest live proxy chain that exercises scheduler dependency
release across Qwen-shaped task bodies. It runs
`qwen_attention_qkv -> qwen_attention_o -> qwen_logits`, validates mutable
KV writeback plus final logits copy-back, and reuses one prepared callable
across repeated `run_prepared` submissions. It still remains controlled proxy
evidence rather than full Qwen model execution.

