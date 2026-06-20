# vLLM Remote H200 256K Needle Correctness Probe

This note records a bounded remote H200 vLLM synthetic needle correctness
probe for `deepseek-ai/DeepSeek-V4-Flash` with `--max-model-len 262144`. It
extends the prior 256K long-prompt response-contract gate by checking whether
the generated output contains an exact synthetic expected answer that appears
once inside the long prompt.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-256k-needle-correctness-probe/`.

## Probe Surface

The repo-owned needle correctness probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends one non-streaming
`/v1/completions` request near a 256K prompt-token budget, validates response
structure and usage accounting, checks exact expected-answer containment,
emits structured JSON, terminates the server process group, and reports
remaining process-group PIDs.

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
generated output contains the exact expected needle answer
short synthetic generated output is recorded when within review-safe bound
usage prompt/completion/total token fields are internally consistent when returned
usage.prompt_tokens matches measured prompt tokens when available
usage.completion_tokens within request max_tokens
usage.total_tokens >= usage.prompt_tokens + usage.completion_tokens
raw prompt text is not recorded
raw request payload is not recorded
token ID arrays are not recorded
logprob values are not recorded
server process group cleanup leaves no remaining PIDs
```

## Resource Plan

The remote source tree was refreshed with `--sync` from local source commit
`ed4c3e48978872b372e347998300ae5b6663b2da` plus this PR's uncommitted probe
changes, preserving the complete ignored artifact directory and
`.venv-vllm-probe`. Remote git metadata was not used as evidence after the
sync.

Remote tooling and package versions:

```text
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
python: 3.12.3
```

The selected physical GPUs were 1 and 7, exposed to vLLM as exactly two
visible devices:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
```

The fixed local port was:

```text
127.0.0.1:28143
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28143/health
GET http://127.0.0.1:28143/v1/models
POST http://127.0.0.1:28143/v1/completions
```

The dry-run preflight emitted `status=planned` for
`target_prompt_tokens=255800`, `max_tokens=64`, and expected answer
`PTO_NEEDLE_256K_CONTEXT_OK_28143`, with `generation_attempted=false`,
`prompt_sent=false`, `prompt_text_recorded=false`, and
`payload_recorded=false`.

## Passing Needle Correctness Probe

Core probe command:

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
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143
```

Result:

```text
status: passed
server_host: 127.0.0.1
server_port: 28143
elapsed_seconds: 159.508
PROBE_EXIT_STATUS=0
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28143 \
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
target_prompt_tokens: 255800
actual_prompt_tokens: 255799
prompt_chars: 1230965
expected_answer: PTO_NEEDLE_256K_CONTEXT_OK_28143
needle_occurrences: 1
max_tokens: 64
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
generated_text_length_chars: 269
usage.prompt_tokens: 255799
usage.completion_tokens: 64
usage.total_tokens: 255863
expected_answer_contained: passed
generated_text_recorded: short_synthetic_output
```

Exact short generated output recorded by the synthetic probe follows. The
interpretive wording in this block is model output only; the repo claim is
limited to exact expected-answer containment.

````text
 PTO_NEEDLE_256K_CONTEXT_OK_28143
```

The model correctly extracts the needle value from the middle of the context. This demonstrates that the vLLM-based local-only probe works correctly with the 256K context length.

## Summary

The vLLM-based local-only needle probe
````

## Shutdown Behavior

The probe terminated the server process group after validation:

```text
terminated: true
killed: false
remaining_process_group_pids: []
cleanup.status: passed
```

## Interpretation

The remote H200 vLLM environment served one bounded near-256K synthetic
needle request for `deepseek-ai/DeepSeek-V4-Flash` from the complete
repo-relative artifacts under this explicit two-GPU boundary:

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
target_prompt_tokens=255800
actual_prompt_tokens=255799
max_tokens=64
```

The server bound to `127.0.0.1`, returned HTTP 200 from `/health`, returned
the served model with `max_model_len=262144` from `/v1/models`, returned HTTP
200 from one non-streaming `/v1/completions` request, exposed the expected
review-safe response structure, reported internally consistent usage
accounting, generated short synthetic output containing the exact expected
answer `PTO_NEEDLE_256K_CONTEXT_OK_28143`, and shut down with no remaining
process-group PIDs reported by the probe. This is a synthetic needle
retrieval correctness gate only.

## Non-Claims

- This is not general generated-text correctness evidence.
- This is not semantic correctness evidence.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.
