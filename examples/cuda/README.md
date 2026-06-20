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

## DeepSeek V4 Flash vLLM 64K Context Health Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28132 \
  --server-log tmp/vllm-64k-context-health-probe/server-28132.log \
  --max-model-len 65536 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This reuses the local-only server-health probe with a larger
`--max-model-len` and still checks only `/health` and `/v1/models`; it does
not send a long prompt or run generation. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_64k_context_health_probe.md`:
the server returned HTTP 200 from both checked endpoints and the model list
reported `max_model_len=65536`. This is a 64K server-health/model-list
capacity gate only, not generated-text correctness, tokenizer semantics,
prompt correctness, 256K context, throughput, latency, production readiness,
or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 128K Context Health Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28133 \
  --server-log tmp/vllm-128k-context-health-probe/server-28133.log \
  --max-model-len 131072 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This reuses the local-only server-health probe with a larger
`--max-model-len` and still checks only `/health` and `/v1/models`; it does
not send a long prompt or run generation. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_128k_context_health_probe.md`:
the server returned HTTP 200 from both checked endpoints and the model list
reported `max_model_len=131072`. This is a 128K server-health/model-list
capacity gate only, not generated-text correctness, tokenizer semantics,
prompt correctness, 256K context, throughput, latency, production readiness,
or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Context Health Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28134 \
  --server-log tmp/vllm-256k-context-health-probe/server-28134.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This reuses the local-only server-health probe with the model's configured
262,144-token length and still checks only `/health` and `/v1/models`; it does
not send a long prompt or run generation. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_256k_context_health_probe.md`:
the server returned HTTP 200 from both checked endpoints and the model list
reported `max_model_len=262144`. This is a 256K server-health/model-list
capacity gate only, not long-prompt behavior, generated-text correctness,
tokenizer semantics, prompt correctness, throughput, latency, production
readiness, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Long-Prompt Admission Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_admission_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28135 \
  --server-log tmp/vllm-long-prompt-admission-probe/server-28135.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 1 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 16K prompt-token budget, records only
review-safe shape/accounting fields, and terminates the server process group.
The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_long_prompt_admission_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=15995`, and cleanup reported no remaining process-group PIDs.
This is long-prompt admission evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 70m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28136 \
  --server-log tmp/vllm-long-prompt-response-contract-probe/server-28136.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 16K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=15995`, `completion_tokens=4`, and `total_tokens=15999`, and
cleanup reported no remaining process-group PIDs. This is long-prompt
response-contract evidence only, not generated-text correctness, tokenizer
semantic correctness, prompt semantic correctness, token identity, logprob,
stop-token, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Chat Exact Canary Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_exact_canary_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28149 \
  --server-log tmp/vllm-chat-exact-canary-probe/server-28149.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --max-tokens 16 --temperature 0.0 --top-p 1.0 \
  --seed 0 --expected-answer PTO_CHAT_EXACT_CANARY_28149
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/chat/completions` request,
and requires the narrowly normalized assistant content to exactly equal the
expected canary string. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_exact_canary_probe.md`.
This is a bounded OpenAI-compatible chat-completions exact-output canary, not
general generated-text correctness, semantic correctness, long-prompt chat
behavior, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Long-Prompt Warmup/Follow-Up Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 75m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_warmup_followup_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28137 \
  --server-log tmp/vllm-long-prompt-warmup-followup-probe/server-28137.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --log-settle-seconds 2
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one labeled warmup synthetic
non-streaming `/v1/completions` request near a 16K prompt-token budget, sends
one labeled follow-up request with the same request shape, validates
review-safe response-contract fields for both responses, records generated
text lengths only, and terminates the server process group. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_long_prompt_warmup_followup_probe.md`:
both requests returned HTTP 200 with one response choice, both responses
reported `prompt_tokens=15995`, `completion_tokens=4`, and
`total_tokens=15999`, and cleanup reported no remaining process-group PIDs.
This is long-prompt warmup/follow-up response-contract evidence only, not
generated-text correctness, tokenizer semantic correctness, prompt semantic
correctness, token identity, logprob, stop-token, throughput, latency,
production readiness, broad determinism, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM 32K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28138 \
  --server-log tmp/vllm-32k-long-prompt-response-contract-probe/server-28138.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 32000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 32K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_32k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=32000`, `completion_tokens=4`, and `total_tokens=32004`, and
cleanup reported no remaining process-group PIDs. This is 32K long-prompt
response-contract evidence only, not generated-text correctness, tokenizer
semantic correctness, prompt semantic correctness, token identity, logprob,
stop-token, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 64K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 90m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28139 \
  --server-log tmp/vllm-64k-long-prompt-response-contract-probe/server-28139.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 64000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 64K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_64k_long_prompt_response_contract_probe.md`:
the request
returned HTTP 200 with one response choice, usage reported
`prompt_tokens=63999`, `completion_tokens=4`, and `total_tokens=64003`, and
cleanup reported no remaining process-group PIDs. This is 64K long-prompt
response-contract evidence only, not generated-text correctness, tokenizer
semantic correctness, prompt semantic correctness, token identity, logprob,
stop-token, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 128K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28140 \
  --server-log tmp/vllm-128k-long-prompt-response-contract-probe/server-28140.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 900 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 128000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 128K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_128k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=127997`, `completion_tokens=4`, and `total_tokens=128001`,
and cleanup reported no remaining process-group PIDs. This is 128K
long-prompt response-contract evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 192K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28141 \
  --server-log tmp/vllm-192k-long-prompt-response-contract-probe/server-28141.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1200 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 192000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 192K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_192k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=191995`, `completion_tokens=4`, and `total_tokens=191999`,
and cleanup reported no remaining process-group PIDs. This is 192K
long-prompt response-contract evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28142 \
  --server-log tmp/vllm-256k-long-prompt-response-contract-probe/server-28142.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 256000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 256K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_256k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=256004`, `completion_tokens=4`, and `total_tokens=256008`,
and cleanup reported no remaining process-group PIDs. This is 256K
long-prompt response-contract evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Needle Correctness Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28143 \
  --server-log tmp/vllm-256k-needle-correctness-probe/server-28143.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143 \
  --match-mode contains
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 256K prompt-token budget, and reports
`status=passed` only when the short generated output satisfies the selected
match mode. The default `contains` mode preserves the existing containment
gate. The remote H200 containment evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_256k_needle_correctness_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=255799`, `completion_tokens=64`, and `total_tokens=255863`,
the generated output contained `PTO_NEEDLE_256K_CONTEXT_OK_28143`, and cleanup
reported no remaining process-group PIDs. This is synthetic needle retrieval
correctness evidence only, not general generated-text correctness, semantic
correctness, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Needle Exact-Output Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28144 \
  --server-log tmp/vllm-256k-needle-exact-output-probe/server-28144.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143 \
  --match-mode exact
```

Exact mode strips leading/trailing whitespace, strips one surrounding
Markdown code fence only when the whole output is fenced, and then compares
the normalized generated output exactly to the expected synthetic answer. It
does not remove explanatory sentences, punctuation, unmatched Markdown fences,
or unrelated tokens. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_256k_needle_exact_output_failure.md`:
the request returned HTTP 200 with one response choice, but strict exact mode
failed because the normalized output retained an unmatched closing Markdown
fence after `PTO_NEEDLE_256K_CONTEXT_OK_28143`. This is a strict synthetic
exact-output failure record only, not general generated-text correctness,
semantic correctness, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

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

## DeepSeek V4 Flash vLLM Warmup-Shape Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_warmup_shape_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28126 \
  --server-log tmp/vllm-warmup-shape-probe/server-28126.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --log-settle-seconds 2
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one labeled warmup `/v1/completions` request, sends one
follow-up same-shape `/v1/completions` request, validates the same structural
response contract for both responses, counts selected Triton JIT warning
strings in server-log windows around the two requests, and terminates the
server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_warmup_shape_probe.md`. This is
warmup-shape observation evidence, not generated-text correctness, tokenizer
semantics, prompt correctness, 256K context, throughput, latency, production
readiness, warmup-eliminates-JIT evidence, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM Request-Shape Variation Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_request_shape_variation_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28127 \
  --server-log tmp/vllm-request-shape-variation-probe/server-28127.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --variation-max-tokens 8 --log-settle-seconds 2
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one known warmup `/v1/completions` request, sends one
same-shape follow-up request to preserve the prior baseline inside the run,
sends one bounded variation request with a different prompt and
`max_tokens=8`, validates the same structural response contract for all three
responses, counts selected Triton JIT warning strings by request window, and
terminates the server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_request_shape_variation_probe.md`.
This is request-shape variation observation evidence, not generated-text
correctness, tokenizer semantics, prompt correctness, 256K context,
throughput, latency, production readiness, broad warmup-eliminates-JIT
evidence, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Serving-Semantics Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_serving_semantics_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28128 \
  --server-log tmp/vllm-serving-semantics-probe/server-28128.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --max-tokens 8 --temperature 0.0 --top-p 1.0 \
  --seed 0 --log-settle-seconds 2
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends two identical bounded deterministic `/v1/completions`
requests, validates the structural response contract for both responses,
then compares response observations: completion text digest, text length,
`finish_reason`, and usage accounting. The generated text is not recorded or
judged for correctness. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_serving_semantics_probe.md`.
This is a bounded serving-semantics observation, not generated-text
correctness, tokenizer semantics, prompt correctness, 256K context,
throughput, latency, production readiness, broad determinism evidence, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Logprobs-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28129 \
  --server-log tmp/vllm-logprobs-contract-probe/server-28129.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --logprobs-contract --logprobs 1 --prompt-logprobs 1
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/completions` request with
explicit `logprobs=1` and `prompt_logprobs=1`, validates the base structural
response contract, then checks only logprob response shape: completion
logprob list lengths match `usage.completion_tokens`, and
`choice.prompt_logprobs` is list-valued and bounded by
`usage.prompt_tokens`. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_logprobs_contract_probe.md`.
This is a bounded logprobs response-shape observation, not generated-text
correctness, tokenizer semantics, prompt correctness, token identity or
logprob value correctness, 256K context, throughput, latency, production
readiness, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Echo-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28130 \
  --server-log tmp/vllm-echo-contract-probe/server-28130.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 1 --temperature 0.0 --top-p 1.0 \
  --seed 0 --echo-contract
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/completions` request with
explicit `echo=true`, validates the base structural response contract, then
checks only echo response shape: the request prompt is string-valued and the
response text starts with that prompt. The remote H200 evidence is recorded
in `docs/in_progress/nvidia_backend/vllm_remote_echo_contract_probe.md`. This
is a bounded echo response-shape observation, not generated-text correctness,
tokenizer semantics, prompt correctness, token identity or logprob value
correctness, 256K context, throughput, latency, production readiness, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Stop-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28131 \
  --server-log tmp/vllm-stop-contract-probe/server-28131.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --stop-contract
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/completions` request with
explicit `stop`, `stop_token_ids`, and `include_stop_str_in_output=false`,
validates the base structural response contract, and checks only that the
explicit stop fields were carried in the accepted request. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_stop_contract_probe.md`. This is
bounded stop-field acceptance and response-contract evidence, not stop-trigger
evidence, generated-text correctness, tokenizer semantics, prompt
correctness, token identity or stop-token semantic correctness, 256K context,
throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.
