# vLLM Remote H200 Chat 256K Needle Streaming Position Sweep Probe

This note records one bounded OpenAI-compatible streaming chat-completions
synthetic needle position-sweep probe for
`deepseek-ai/DeepSeek-V4-Flash`. The gate checks whether early, middle, and
late needle placements each satisfy the strict chat 256K streaming
exact-output contract in one local-only vLLM server lifecycle.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-chat-256k-needle-stream-position-sweep-probe/`.

## Probe Surface

The repo-owned streaming position-sweep entry point reuses the chat 256K
needle prompt builder, lifecycle, readiness contract, SSE parser, streaming
safety checks, and exact normalization. It posts one streaming request for
each requested position:

````text
endpoint: /v1/chat/completions
stream: true
needle_position_sweep: early,middle,late
expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156
stop: ["\n```"]
match_mode: exact
````

Each position must parse SSE content events, receive terminal `[DONE]`,
record a final streaming `finish_reason`, and narrowly normalize the
assembled assistant content to exactly the expected answer.

## Remote Preflight

The preserved remote vLLM probe environment should be reused. Refresh the
source tree with `--sync`, which excludes `.venv`, `.venv-*`, `build`, `tmp`,
Python caches, and pytest caches, preserving the existing remote vLLM probe
venv and repo-relative model artifact directory.

The bounded diagnostic used the selected physical GPUs 1 and 7:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
GPU 1: NVIDIA H200 NVL
GPU 7: NVIDIA H200 NVL
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
python: 3.12.3
```

## Streaming Position Sweep Diagnostic

Core probe command shape:

````bash
REMOTE_PTO_CU=<preserved-remote-env-artifact> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc '<bounded command>'

CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_position_sweep_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28156 \
  --server-log tmp/vllm-chat-256k-needle-stream-position-sweep-probe/server-28156.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156 \
  --needle-position-sweep early,middle,late
````

Result:

```text
status: passed
server_host: 127.0.0.1
server_port: 28156
PROBE_EXIT_STATUS=0
generation_attempted: true
prompt_sent: true
stream_events_received: true
elapsed_seconds: 186.952
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
max_tokens=64
temperature=0.0
top_p=1.0
seed=0
expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156
actual_prompt_tokens: not_available
prompt_chars: 1233756
prompt_unit_chars: 82
needle_occurrences: 1
needle_position_sweep: early,middle,late
message_count: 2
message_roles: system,user
match_mode: exact
stop_sequences_configured: true
stop: ["\n```"]
stream: true
n: 1
prompt_text_recording: false
payload_recording: false
assistant_content_recording: false
```

Position sweep result:

```text
status: passed
positions_completed: 3
passed_positions: 3
failed_positions: 0
position: early
early HTTP status: 200
early event_count: 22
early content_chunk_count: 19
early done_seen: true
early finish_reason: stop
early usage: not_returned
early normalized_output_equals_expected: true
early normalized_output_length_chars: 42
early expected_answer_exact: passed
position: middle
middle HTTP status: 200
middle event_count: 22
middle content_chunk_count: 19
middle done_seen: true
middle finish_reason: stop
middle usage: not_returned
middle normalized_output_equals_expected: true
middle normalized_output_length_chars: 42
middle expected_answer_exact: passed
position: late
late HTTP status: 200
late event_count: 21
late content_chunk_count: 19
late done_seen: true
late finish_reason: stop
late usage: not_returned
late normalized_output_equals_expected: true
late normalized_output_length_chars: 42
late expected_answer_exact: passed
```

Cleanup:

```text
cleanup.status: passed
terminated: true
killed: false
returncode_after_cleanup: 0
remaining_process_group_pids: []
```

## Interpretation

This is one local-only two-H200 streaming `/v1/chat/completions` synthetic
needle position-sweep pass under the recorded boundary. The run sent exactly
one streaming request per requested position in one vLLM server lifecycle,
parsed server-sent events for every position, required terminal `[DONE]` and
final `finish_reason` for every position, and required the narrowly
normalized assembled assistant content to exactly equal
`PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156` for every position.

Usage is recorded as the returned usage object when vLLM includes it, or as
`not_returned`; this does not weaken the strict exact comparator.

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
raw streaming chunk content is not recorded
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
