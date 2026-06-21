# NVIDIA Backend Work Preparation

This note keeps the next DeepSeek-V4-Flash review gate explicit without
claiming serving success.

## DeepSeek V4 Flash Gate Order

Before attempting any real shard download, run the acquisition preflight:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  examples/cuda/deepseek_v4_flash_weight_acquisition_preflight.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --metadata tmp/sources/model-metadata/deepseek-ai-DeepSeek-V4-Flash.json \
  --download-root tmp \
  --reserve-bytes 10737418240 \
  --capacity-multiplier 1.1 \
  --require-capacity
```

Proceed to shard acquisition only when `can_attempt_download` is true or the
manifest is already complete. Proceed to model-load planning only when
`can_attempt_model_load` is true. This is not serving evidence.
This is not model-load evidence. This is not DeepSeek correctness evidence.
Use `--fetch-hf-metadata` only when the host has outbound Hugging Face access;
otherwise pass a review-safe metadata JSON with `--metadata`.

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
