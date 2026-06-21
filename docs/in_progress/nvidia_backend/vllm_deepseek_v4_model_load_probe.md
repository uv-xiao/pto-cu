# vLLM DeepSeek V4 Model-Load Probe

This note records the repo-owned pre-serving model-load gate for
`deepseek-ai/DeepSeek-V4-Flash`. The gate is intentionally narrower than
serving: it checks local artifact readiness, installed vLLM, CUDA visibility,
and then constructs `vllm.LLM` only when every required precondition is
available.

## Probe Surface

Tracked files:

- `examples/cuda/vllm_deepseek_v4_model_load_probe.py`
- `tests/ut/py/test_vllm_deepseek_v4_model_load_probe.py`

The probe composes the existing artifact inspection logic from
`examples/cuda/vllm_deepseek_v4_artifact_probe.py`, then reports separate
structured blocks for:

- `artifact_probe`
- `vllm_probe`
- `cuda_probe`
- `llm_kwargs`
- `runtime_versions`
- `gpu_memory_before` and `gpu_memory_after` when `nvidia-smi` is available

Missing artifacts, missing vLLM, and missing CUDA are structured skips by
default. Use `--require-artifacts`, `--require-vllm`, and `--require-cuda`
when those missing prerequisites should fail the command.

## Command Shape

Use a repo-relative artifact directory and an explicit GPU boundary:

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

For local planning without an engine construction attempt, add `--dry-run`.
The dry-run mode still reports the artifact, vLLM, CUDA, and planned vLLM
argument surfaces, but it is not model-load evidence.

## Interpretation

A passing run means the process constructed `vllm.LLM`, observed the returned
object class, optionally observed `llm_engine`, and called engine shutdown when
that method is exposed. It does not start an HTTP server.

The probe does not run generation.

A missing-shard or missing-vLLM run is not model-load evidence. A missing-CUDA
run is also not model-load evidence. Those outcomes only prove that the gate
blocked before a model-load attempt.

H200 was not rerun for this child slice. The existing historical remote note
`docs/in_progress/nvidia_backend/vllm_remote_model_load_probe.md` remains the
record for the earlier two-H200 run and its exact resource boundary.

## Non-Claims

- This is not DeepSeek-V4-Flash serving success.
- This is not correct output text.
- This is not long-context correctness.
- This is not server health, OpenAI API, latency, throughput, or production
  readiness evidence.
- This is not simpler-nv/vLLM kernel integration evidence.
- Missing-shard, missing-vLLM, missing-CUDA, and dry-run outputs are not
  model-load evidence.
