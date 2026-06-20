# vLLM Remote H200 Long-Prompt Warmup/Follow-Up Probe

This note records a bounded remote H200 vLLM long-prompt warmup/follow-up
probe for `deepseek-ai/DeepSeek-V4-Flash` with `--max-model-len 262144`. It
extends the prior 16K-ish long-prompt response-contract gate by validating
two consecutive same-shape completion requests in one local-only server
lifecycle, without judging generated text.

Raw command output is kept under the gitignored local directory
`tmp/vllm-long-prompt-warmup-followup-probe/`.

## Probe Surface

The repo-owned long-prompt warmup/follow-up probe starts `vllm serve`, binds
only to `127.0.0.1`, checks `/health` and `/v1/models`, sends one labeled
`warmup` non-streaming `/v1/completions` request near a 16K prompt-token
budget, sends one labeled `followup` request with the same request shape,
validates response structure and usage accounting for both responses, emits
structured JSON, terminates the server process group, and reports remaining
process-group PIDs.

Contract checks:

```text
HTTP 200 from /health
HTTP 200 from /v1/models
model list includes served model and max_model_len=262144
HTTP 200 from warmup non-streaming /v1/completions request
HTTP 200 from followup non-streaming /v1/completions request
top-level completion responses are JSON objects
response model fields match served model when returned
exactly one response choice object per response
first choices expose text and finish_reason fields
generated text lengths are recorded without generated text contents
usage prompt/completion/total token fields are internally consistent when returned
usage.prompt_tokens matches measured prompt tokens when available
usage.completion_tokens within request max_tokens
usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens
raw prompt text is not recorded
raw generated text is not recorded
server process group cleanup leaves no remaining PIDs
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the probe. The
ignored repo-relative artifact directory and checkout-local `.venv-vllm-probe`
were preserved.

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
127.0.0.1:28137 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28137/health
GET http://127.0.0.1:28137/v1/models
POST http://127.0.0.1:28137/v1/completions  # warmup
POST http://127.0.0.1:28137/v1/completions  # followup
```

Timeouts:

```text
outer timeout: 75m
readiness timeout: 2700s
request timeout: 600s per completion request
poll interval: 10s
cleanup timeout: 60s
```

## Passing Warmup/Follow-Up Probe

Core probe command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'RUN_DIR=tmp/vllm-long-prompt-warmup-followup-probe
RUN_LOG="$RUN_DIR/server-28137.log"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 75m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_long_prompt_warmup_followup_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28137 \
  --server-log "$RUN_LOG" \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --log-settle-seconds 2'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 120.038
server_host: 127.0.0.1
server_port: 28137
prompt_sent: true
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28137 \
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
/health: HTTP 200 after 11 polling attempts
/v1/models: HTTP 200
warmup /v1/completions: HTTP 200
followup /v1/completions: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 262144
```

Request accounting for both requests:

```text
target_prompt_tokens: 16000
actual_prompt_tokens: 15995
tokenizer_accounting: transformers.AutoTokenizer local encode
prompt_chars: 104669
max_tokens: 4
temperature: 0.0
top_p: 1.0
seed: 0
stream: false
echo: false
logprobs: false
prompt_text_recorded: false
payload_recorded: false
```

Warmup response shape and accounting:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
finish_reason: length
generated_text_length_chars: 23
usage.prompt_tokens: 15995
usage.completion_tokens: 4
usage.total_tokens: 15999
raw_generated_text_recorded: false
```

Follow-up response shape and accounting:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
finish_reason: length
generated_text_length_chars: 23
usage.prompt_tokens: 15995
usage.completion_tokens: 4
usage.total_tokens: 15999
raw_generated_text_recorded: false
```

Recorded response shape for both responses:

```text
top-level keys:
  choices, created, id, kv_transfer_params, model, object, service_tier,
  system_fingerprint, usage
first choice keys:
  finish_reason, index, logprobs, prompt_logprobs, prompt_token_ids,
  routed_experts, stop_reason, text, token_ids
usage keys:
  completion_tokens, prompt_tokens, prompt_tokens_details, total_tokens
```

vLLM server log evidence:

```text
Using max model len 262144
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Available KV cache memory: 32.08 GiB
GPU KV cache size: 2,175,276 tokens
Maximum concurrency for 262,144 tokens per request: 8.30x
Starting vLLM server on http://127.0.0.1:28137
Route: /health, Methods: GET
Route: /v1/models, Methods: GET
Route: /v1/completions, Methods: POST
GET /health HTTP/1.1" 200 OK
GET /v1/models HTTP/1.1" 200 OK
POST /v1/completions HTTP/1.1" 200 OK
POST /v1/completions HTTP/1.1" 200 OK
Shutting down
Application shutdown complete.
```

The server log also recorded selected Triton JIT warning strings before the
first completion response. That is a run observation only, not latency,
throughput, or broad warmup behavior evidence.

## Shutdown Behavior

The probe terminated the server process group after both contract validations:

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

The remote H200 vLLM environment served two consecutive same-shape bounded
16K-ish prompt completion requests for
`deepseek-ai/DeepSeek-V4-Flash` from the complete repo-relative artifacts
under this explicit two-H200 boundary:

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
max_tokens=4
```

The server bound to `127.0.0.1`, returned HTTP 200 from `/health`, returned
the served model with `max_model_len=262144` from `/v1/models`, returned HTTP
200 from both non-streaming `/v1/completions` requests, exposed the expected
response structure for both responses, reported internally consistent usage
accounting for both responses, and shut down with no remaining process-group
PIDs reported by the probe. This is a long-prompt warmup/follow-up
response-contract gate only.

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

Run a bounded follow-up gate that either increases the long-prompt budget
beyond this 16K-ish class or varies one request shape after the same 16K-ish
warmup, while preserving the same review-safe response/accounting contract and
without adding generated-text correctness claims.
