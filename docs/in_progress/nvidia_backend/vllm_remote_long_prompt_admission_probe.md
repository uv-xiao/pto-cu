# vLLM Remote H200 Long-Prompt Admission Probe

This note records a bounded remote H200 vLLM long-prompt admission probe for
`deepseek-ai/DeepSeek-V4-Flash` with `--max-model-len 262144`. It reuses the
complete repo-relative artifact directory and local-only server boundary from
the 256K endpoint-only health gate, then sends exactly one bounded completion
request.

Raw command output is kept under the gitignored local directory
`tmp/vllm-long-prompt-admission-probe/`.

## Probe Surface

The repo-owned long-prompt admission probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends one non-streaming
`/v1/completions` request, emits structured JSON, terminates the server
process group, and reports remaining process-group PIDs.

Contract checks:

```text
HTTP 200 from /health
HTTP 200 from /v1/models
HTTP 200 from one non-streaming /v1/completions request
exactly one response choice when HTTP 200 returns
usage fields recorded when returned
raw prompt text is not recorded
raw generated text is not recorded
server process group cleanup leaves no remaining PIDs
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the run. The ignored
repo-relative artifact directory and checkout-local `.venv-vllm-probe` were
preserved.

Remote tooling and package versions:

```text
GPU: 2 selected NVIDIA H200 NVL devices from an 8-GPU host
python: 3.12.3
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
```

The selected physical GPUs were 1 and 7, exposed to vLLM as exactly two
visible devices:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
```

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116670 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28135 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28135/health
GET http://127.0.0.1:28135/v1/models
POST http://127.0.0.1:28135/v1/completions
```

Timeouts:

```text
outer timeout: 65m
readiness timeout: 2700s
request timeout: 600s
poll interval: 10s
cleanup timeout: 60s
```

## Passing Admission Probe

Core probe command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'RUN_DIR=tmp/vllm-long-prompt-admission-probe
RUN_LOG="$RUN_DIR/server-28135.log"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_long_prompt_admission_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28135 \
  --server-log "$RUN_LOG" \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 1 \
  --temperature 0.0 --top-p 1.0 --seed 0'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 126.517
server_host: 127.0.0.1
server_port: 28135
prompt_sent: true
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28135 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer-mode deepseek_v4 \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

Endpoint results:

```text
/health: HTTP 200 after 12 polling attempts
/v1/models: HTTP 200
/v1/completions: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 262144
```

Request accounting:

```text
target_prompt_tokens: 16000
actual_prompt_tokens: 15995
tokenizer_accounting: transformers.AutoTokenizer local encode
prompt_chars: 104669
max_tokens: 1
temperature: 0.0
top_p: 1.0
seed: 0
stream: false
echo: false
logprobs: false
prompt_text_recorded: false
payload_recorded: false
```

Response shape and accounting:

```text
choice_count: 1
first_choice_keys:
  finish_reason, index, logprobs, prompt_logprobs, prompt_token_ids,
  routed_experts, stop_reason, text, token_ids
usage_keys:
  completion_tokens, prompt_tokens, prompt_tokens_details, total_tokens
usage.prompt_tokens: 15995
usage.completion_tokens: 1
usage.total_tokens: 15996
raw_generated_text_recorded: false
```

vLLM server log evidence:

```text
Using max model len 262144
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Available KV cache memory: 32.08 GiB
GPU KV cache size: 2,175,276 tokens
Maximum concurrency for 262,144 tokens per request: 8.30x
Starting vLLM server on http://127.0.0.1:28135
Route: /v1/completions, Methods: POST
GET /health HTTP/1.1" 200 OK
GET /v1/models HTTP/1.1" 200 OK
POST /v1/completions HTTP/1.1" 200 OK
Shutting down
Application shutdown complete.
```

## Shutdown Behavior

The probe terminated the server process group after the endpoint and
completion checks:

```text
terminated: true
killed: false
returncode_after_cleanup: 0
remaining_process_group_pids: []
cleanup.status: passed
```

The immediate post-run selected-GPU snapshot matched the pre-run baseline:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116670 MiB free / 143771 MiB total
```

## Interpretation

The remote H200 vLLM environment accepted and completed one bounded
long-prompt completion request for `deepseek-ai/DeepSeek-V4-Flash` from the
complete repo-relative artifacts under this explicit two-H200 boundary:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
max_model_len=262144
dtype=bfloat16
quantization=deepseek_v4_fp8
kv_cache_dtype=fp8
gpu_memory_utilization=0.78
enforce_eager=true
distributed_executor_backend=mp
target_prompt_tokens=16000
actual_prompt_tokens=15995
max_tokens=1
```

The server bound to `127.0.0.1`, returned HTTP 200 from `/health`, returned
the served model with `max_model_len=262144` from `/v1/models`, returned HTTP
200 from one non-streaming `/v1/completions` request, and shut down with no
remaining process-group PIDs reported by the probe. This is a long-prompt
admission gate only.

## Non-Claims

- This is not generated-text correctness evidence.
- This is not tokenizer semantic correctness or prompt semantic correctness
  evidence.
- This is not token identity, logprob, or stop-token evidence.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.

## Recommended Next Gate

The next PR-sized gate can stay under the same local-only vLLM boundary and
probe a bounded long-prompt response-contract variant, still without judging
generated text. A separate later gate can decide whether to raise the prompt
budget beyond this 16K admission check.
