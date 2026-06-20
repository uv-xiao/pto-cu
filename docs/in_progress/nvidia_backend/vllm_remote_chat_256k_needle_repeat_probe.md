# vLLM Remote H200 Chat 256K Needle Exact Repeat Probe

This note records a bounded same-server OpenAI-compatible chat-completions
synthetic needle exact-repeat probe for `deepseek-ai/DeepSeek-V4-Flash`.
The goal was to check whether one local-only vLLM server lifecycle can satisfy
the strict chat 256K needle exact comparator for exactly two identical
non-streaming `/v1/chat/completions` requests.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-chat-256k-needle-repeat-probe/`.

## Probe Surface

The repo-owned chat 256K needle exact probe now accepts repeat execution
through shared review-safe repeat helpers. The dedicated repeat entry point
sets the PR-sized boundary defaults:

```text
endpoint: /v1/chat/completions
repeat_count: 2
expected_answer: PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152
stop_sequence: \n```
```

Each attempt sends the same bounded non-streaming chat request, runs the
existing response-contract checks, and applies the same narrow exact-match
normalization used by the single-request chat 256K needle gate.

Repeat summaries are review-safe. They include attempt index, endpoint, HTTP
status, finish reason, exact-check result, normalized-output equality boolean,
normalized-output length, and usage token counts. They do not include raw
prompt text, raw request payloads, raw assistant/generated text, token ID
arrays, logprob values, generated-text digests, model artifact contents,
private absolute paths, hostnames, usernames, or non-loopback URLs.

## Remote Preflight

The preserved remote vLLM probe environment was reused. The source tree was
refreshed with `--sync`, which excludes `.venv`, `.venv-*`, `build`, `tmp`,
Python caches, and pytest caches, preserving the existing remote vLLM probe
venv and repo-relative model artifact directory.

The selected physical GPUs for the run were 1 and 7:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
GPU 1: NVIDIA H200 NVL
GPU 7: NVIDIA H200 NVL
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
```

## Passing Repeat Diagnostic

Core probe command shape:

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_chat_256k_needle_repeat_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28152 \
  --server-log tmp/vllm-chat-256k-needle-repeat-probe/server-28152.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152
```

Result:

```text
status: passed
server_host: 127.0.0.1
server_port: 28152
PROBE_EXIT_STATUS=0
generation_attempted: true
elapsed_seconds: 155.889
repeat_count: 2
attempts_completed: 2
passed_attempts: 2
failed_attempts: 0
```

Readiness and request controls:

```text
/health HTTP status: 200
/v1/models HTTP status: 200
model_list includes: deepseek-ai/DeepSeek-V4-Flash
model_list max_model_len: 262144
endpoint: /v1/chat/completions
max_model_len=262144
tensor_parallel_size=2
dtype=bfloat16
quantization=deepseek_v4_fp8
kv_cache_dtype=fp8
gpu_memory_utilization=0.78
distributed_executor_backend=mp
enforce_eager=true
target_prompt_tokens=255800
actual_prompt_tokens: not_available
usage.prompt_tokens: 255796
max_tokens=64
temperature=0.0
top_p=1.0
seed=0
expected_answer: PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152
needle_occurrences: 1
needle_position: middle
message_count: 2
message_roles: system,user
match_mode: exact
stop_sequences_configured: true
stop_sequence: \n```
stream: false
n: 1
tokenizer_accounting: transformers.AutoTokenizer fallback encode estimate
prompt_text_recording: false
payload_recording: false
assistant_content_recording: false
```

Repeat attempt summaries:

```text
attempt_index: 1
endpoint: /v1/chat/completions
status: passed
HTTP status: 200
finish_reason: stop
normalized_output_equals_expected: true
normalized_output_length_chars: 36
exact_check: passed
usage.prompt_tokens: 255796
usage.completion_tokens: 19
usage.total_tokens: 255815

attempt_index: 2
endpoint: /v1/chat/completions
status: passed
HTTP status: 200
finish_reason: stop
normalized_output_equals_expected: true
normalized_output_length_chars: 36
exact_check: passed
usage.prompt_tokens: 255796
usage.completion_tokens: 19
usage.total_tokens: 255815
```

Cleanup:

```text
terminated: true
killed: false
returncode_after_cleanup: 0
remaining_process_group_pids: []
```

## Interpretation

This is one local-only two-H200 same-server synthetic
`/v1/chat/completions` needle exact-repeat pass under the recorded boundary.
Both attempts used the same request limits, strict exact comparator, stop
sequence, and expected answer, and both attempts passed.

It does not establish general generated-text correctness, semantic
correctness, tokenizer semantic correctness, token identity, logprob
correctness, broad stop-trigger behavior, broad determinism, latency,
throughput, production readiness, simpler-nv integration, or vLLM kernel
integration.

## Review-Safety Omissions

```text
raw prompt text is not recorded
raw request payload is not recorded
raw generated text is not recorded
token ID arrays are not recorded
logprob values are not recorded
generated-text digests are not recorded
model artifact contents and symlinks are not recorded
private absolute paths are not recorded
hostnames are not recorded
usernames are not recorded
non-loopback URLs are not recorded
```

## Non-Claims

- This is not general generated-text correctness evidence.
- This is not semantic correctness evidence.
- This is not tokenizer semantic correctness evidence.
- This is not token identity evidence.
- This is not logprob correctness evidence.
- This is not broad stop-trigger or stop-token semantic correctness evidence.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.
