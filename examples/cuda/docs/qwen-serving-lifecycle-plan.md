# CUDA Examples: Qwen Serving Lifecycle Plan

## Qwen Serving Lifecycle Plan

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_serving_lifecycle_plan.py \
  --output-json tmp/cuda-backend/pto-serving-lifecycle/qwen-serving-lifecycle-plan.json
```

Expected output: command exits 0; output JSON records the Qwen3-8B model
shape, KV-cache capacity ladder, weight-binding plan, and persistent-device
task mapping for the MPK and VDCores serving policies.

This is a lifecycle contract artifact, not a full-serving result. It makes the
memory and callable mapping reviewable before tokenizer, safetensors loading,
kernel bodies, decode-loop execution, and viewer-result import are complete.

