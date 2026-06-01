# CUDA Examples: Qwen Unit Math Live

## Qwen Unit Math Live

- Benchmark id: `llm_serving_decode`
- Runtime: `cuda/persistent_device`
- Method id: `pto_persistent_device`

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_unit_math_live.py \
  --device 0 \
  --arch compute_80 \
  --repeat-runs 3 \
  --output-json tmp/cuda-backend/pto-serving-unit-math-live/qwen-unit-math-live.json
```

Expected output: command exits 0 on a CUDA host; output JSON records
status=pass for a Qwen unit-math DAG launched repeatedly through
cuda/persistent_device with one prepared callable.

This is live runtime evidence for the unit math path only. It proves the
RMSNorm, QKV cache writeback, SiLU/SwiGLU, and logits task-body path can be
compiled, scheduled, reused across prepared submissions, and copied back. It
is not a full Qwen decode loop.

