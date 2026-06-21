# NVIDIA Backend Work Preparation

This note keeps the next DeepSeek-V4-Flash review gate explicit without
claiming serving success.

## DeepSeek V4 Flash Gate Order

The local weight manifest can report `preflight_status:
ready_for_model_load` and `next_gate: run_model_load_probe` only after the
artifact directory, weight index, and indexed shards are present.

The next command for that gate is:

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 45m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_model_load_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm --require-cuda \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp \
  --enforce-eager
```

This is a pre-serving model-load gate. It must not be interpreted as
DeepSeek-V4-Flash serving success, generated-text correctness, long-context
correctness, or simpler-nv/vLLM kernel integration evidence.
