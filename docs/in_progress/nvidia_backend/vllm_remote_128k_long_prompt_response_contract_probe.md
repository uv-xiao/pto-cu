# vLLM Remote H200 128K Long-Prompt Response-Contract Probe

This note records a bounded remote H200 vLLM long-prompt response-contract
probe for `deepseek-ai/DeepSeek-V4-Flash` with `--max-model-len 262144`. It
extends the prior 64K long-prompt response-contract gate by raising the
synthetic prompt budget to 128K prompt tokens while preserving the same
review-safe response/accounting contract.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-128k-long-prompt-response-contract-probe/`.

## Probe Surface

The repo-owned long-prompt response-contract probe starts `vllm serve`, binds
only to `127.0.0.1`, checks `/health` and `/v1/models`, sends one
non-streaming `/v1/completions` request near a 128K prompt-token budget,
validates response structure and usage accounting, emits structured JSON,
terminates the server process group, and reports remaining process-group
PIDs.

Contract checks:

```text
HTTP 200 from /health
HTTP 200 from /v1/models
model list includes served model and max_model_len=262144
HTTP 200 from one non-streaming /v1/completions request
top-level completion response is a JSON object
response model field matches served model when returned
exactly one response choice object
first choice exposes text and finish_reason fields
generated text length is recorded without generated text contents
usage prompt/completion/total token fields are internally consistent when returned
usage.prompt_tokens matches measured prompt tokens when available
usage.completion_tokens within request max_tokens
usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens
raw prompt text is not recorded
raw generated text is not recorded
server process group cleanup leaves no remaining PIDs
```

## Resource Plan

The remote source tree was refreshed with `--sync` during preflight from local
source commit `cfc5490c2e62bb65c655d5ccfbe7707ecfea4229`, preserving the
complete ignored artifact directory and `.venv-vllm-probe`. Remote git
metadata was not used as evidence after the sync.

Remote tooling and package versions:

```text
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

The fresh pre-run selected-GPU memory snapshot showed 141773 MiB free on GPU 1
and 116670 MiB free on GPU 7.

The fixed local port was checked as available before the run:

```text
127.0.0.1:28140
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28140/health
GET http://127.0.0.1:28140/v1/models
POST http://127.0.0.1:28140/v1/completions
```

The dry-run preflight emitted `status=planned` for `target_prompt_tokens=128000`
and `max_tokens=4`, with `generation_attempted=false`, `prompt_sent=false`,
`prompt_text_recorded=false`, and `payload_recorded=false`.

## Passing Response-Contract Probe

Core probe command:

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

Result:

```text
status: passed
server_host: 127.0.0.1
server_port: 28140
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28140 \
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
/health: HTTP 200
/v1/models: HTTP 200
/v1/completions: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 262144
```

Request accounting:

```text
target_prompt_tokens: 128000
actual_prompt_tokens: 127997
prompt_chars: 837773
max_tokens: 4
temperature: 0.0
top_p: 1.0
seed: 0
stream: false
echo: false
logprobs: false
prompt_text_recorded: false
payload_recorded: false
tokenizer_accounting: transformers.AutoTokenizer local encode
```

Response shape and accounting:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
finish_reason: length
generated_text_length_chars: 23
usage.prompt_tokens: 127997
usage.completion_tokens: 4
usage.total_tokens: 128001
raw_generated_text_recorded: false
```

## Shutdown Behavior

The probe terminated the server process group after contract validation:

```text
terminated: true
killed: false
remaining_process_group_pids: []
cleanup.status: passed
```

The immediate post-run selected-GPU memory snapshot matched the pre-run
baseline for GPU 1 and GPU 7. A follow-up loopback bind check also reported
port 28140 free after cleanup.

## Interpretation

The remote H200 vLLM environment served one bounded 128K prompt completion
request for `deepseek-ai/DeepSeek-V4-Flash` from the complete repo-relative
artifacts under this explicit two-GPU boundary:

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
target_prompt_tokens=128000
actual_prompt_tokens=127997
max_tokens=4
```

The server bound to `127.0.0.1`, returned HTTP 200 from `/health`, returned
the served model with `max_model_len=262144` from `/v1/models`, returned HTTP
200 from one non-streaming `/v1/completions` request, exposed the expected
response structure, reported internally consistent usage accounting, and shut
down with no remaining process-group PIDs reported by the probe. This is a
128K long-prompt response-contract gate only.

## Non-Claims

- This is not generated-text correctness evidence.
- This is not tokenizer semantic correctness or prompt semantic correctness
  evidence.
- This is not token identity, logprob, or stop-token evidence.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.
