# CUDA Examples

These examples preserve review-facing CUDA backend metadata and provide small
skip-safe probes for current work. Historical benchmark rows do not run fresh
CUDA hardware checks; their A100/H200 measurements remain the `743709f3`
capture documented under `docs/nvidia-backend/history/`.

## Host-Schedule Vector Ops

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/host_schedule_vector_ops.py \
  --describe --op add --n 1024 --arch compute_80
```

Use `--op` to select the evaluated host-schedule ABI shape:
`add`, `mul`, `scale`, `square`, `axpy`, `affine`, `triad`, `quad`,
`generic_args`, or `generic_args4`.

## Persistent Layered-Cross Graph

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_layered_cross.py \
  --describe --n 1024 --arch compute_80 --scheduler-blocks 3
```

This describes the same `graph_descriptor_layered_cross` shape that feeds the
current `743709f3` benchmark gate.

## Persistent MoE Dispatch/Combine

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --output-json tmp/persistent-moe-dispatch-combine-local.json
```

This emits structured JSON for `graph_descriptor_moe_dispatch_combine`: four
expert transform tasks, one weighted combine task, and device-side fan-in
before the combine. Without CUDA tooling or a visible NVIDIA GPU it reports a
skip; with `--require-cuda`, the same skip returns a non-zero exit status.

## DeepSeek V4 Flash Weight Manifest

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/deepseek_v4_flash_weight_manifest.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --metadata tmp/sources/model-metadata/deepseek-ai-DeepSeek-V4-Flash.json \
  --require-complete
```

This checks local gitignored shard presence against
`model.safetensors.index.json`. The completed local artifact evidence is
recorded in
`docs/in_progress/nvidia_backend/deepseek_v4_flash_weight_manifest_complete.md`.
It is not model-load or serving evidence.

## DeepSeek V4 Flash Artifact Probe

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
```

This combines local config/tokenizer/index/shard readiness with the existing
weight-free vLLM DeepSeek V4 import and synthetic config probes. Missing local
artifacts or missing vLLM report structured skips by default; use
`--require-artifacts` or `--require-vllm` to make either condition fail the
command. It does not attempt model load, start a server, or run inference.

The remote H200 readiness slices are recorded in
`docs/in_progress/nvidia_backend/vllm_remote_install_probe.md` and
`docs/in_progress/nvidia_backend/deepseek_v4_flash_serving_readiness.md`.
They record remote H200 reachability and the current serving-readiness
boundary.
The follow-up environment/artifact gate is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_env_artifact_probe.md`: the
remote vLLM import/config probes pass in `.venv-vllm-probe`, while the
artifact gates fail because the repo-relative artifact path contains
metadata/tokenizer files but not the indexed weight shards. The artifact
completion gate is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_artifact_complete.md`: the same
remote vLLM import/config probes pass, and the artifact/manifest gates now
find all 46 indexed shards at the repo-relative artifact path. These gates are
not model-load or serving evidence.

## DeepSeek V4 Flash vLLM Model-Load Probe

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_model_load_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

Run it only under an explicit GPU boundary, for example
`CUDA_VISIBLE_DEVICES=<two ids>` with a matching `--tensor-parallel-size 2`,
and an external timeout. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_model_load_probe.md`: vLLM loaded
all 46 shards and initialized an `LLMEngine` on two H200 GPUs at
`max_model_len=4096`. This is model-load and engine-initialization evidence,
not server health, inference correctness, 256K context, throughput, latency, or
production-readiness evidence.

## DeepSeek V4 Flash vLLM Server Health Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28123 \
  --server-log tmp/vllm-server-health-probe/server-28123.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This starts a local-only OpenAI-compatible vLLM server bound to `127.0.0.1`,
checks `/health` and `/v1/models`, emits structured JSON, and terminates the
server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_server_health_probe.md`: the
server started for `deepseek-ai/DeepSeek-V4-Flash`, returned HTTP 200 from
both checked endpoints, and shut down with no remaining process-group PIDs
reported by the probe. This is server startup and health/model-list evidence,
not generated-text correctness, tokenizer semantics, 256K context,
throughput, latency, production readiness, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM Inference Smoke Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 55m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_inference_smoke_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28124 \
  --server-log tmp/vllm-inference-smoke-probe/server-28124.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --endpoint /v1/completions --prompt Hello --max-tokens 1 \
  --temperature 0.0
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends exactly one bounded completion request by default, records
request limits and response shape, and terminates the server process group.
The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_inference_smoke_probe.md`: the
server returned HTTP 200 from readiness endpoints, returned HTTP 200 from one
`/v1/completions` request with `max_tokens=1`, and shut down with no remaining
process-group PIDs reported by the probe. This is inference-smoke evidence,
not generated-text correctness, tokenizer semantics, prompt correctness, 256K
context, throughput, latency, production readiness, or simpler-nv/vLLM
integration evidence.

## DeepSeek V4 Flash vLLM Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 60m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28125 \
  --server-log tmp/vllm-response-contract-probe/server-28125.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 --seed 0
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming completion request with explicit
sampler settings, validates OpenAI-compatible response structure, and
terminates the server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_response_contract_probe.md`: the
server returned HTTP 200 from readiness endpoints and `/v1/completions`, the
response had exactly one choice, usage token counts were internally
consistent and within the request bound, and cleanup reported no remaining
process-group PIDs. This is response-contract evidence, not generated-text
correctness, tokenizer semantics, prompt correctness, 256K context,
throughput, latency, production readiness, or simpler-nv/vLLM integration
evidence.
