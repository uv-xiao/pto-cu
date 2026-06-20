# vLLM Remote H200 Chat 256K Needle Streaming Truncated Failure Probe

This note records one bounded OpenAI-compatible streaming
chat-completions synthetic needle exact-output failure-mode
characterization for `deepseek-ai/DeepSeek-V4-Flash`. It intentionally uses
`--max-tokens 1`, so the strict exact comparator is expected to fail while
the request, transport, streaming parser, and server cleanup path remain
review-safe.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-chat-256k-needle-stream-truncated-failure-probe/`.

## Probe Surface

The existing repo-owned streaming entry point was reused without adding a
duplicate script:

````text
endpoint: /v1/chat/completions
stream: true
expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_TRUNCATED_OK_28155
stop: ["\n```"]
match_mode: exact
max_tokens=1
````

The streaming parser accepts server-sent events, counts parsed JSON events
and assistant content delta chunks, assembles deltas only in memory for the
strict comparator, and records only review-safe counters, finish reason,
usage state, exact-match status, and cleanup state.

## Remote Preflight

The preserved remote vLLM probe environment was reused. The source tree was
refreshed with `--sync`, which excludes `.venv`, `.venv-*`, `build`, `tmp`,
Python caches, and pytest caches, preserving the existing remote vLLM probe
venv and repo-relative model artifact directory.

The selected physical GPUs for the bounded diagnostic were 1 and 7:

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

## Expected Streaming Truncated-Failure Diagnostic

Core probe command shape:

````bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc '<bounded command that preserves PROBE_EXIT_STATUS>'

RUN_DIR=tmp/vllm-chat-256k-needle-stream-truncated-failure-probe
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28155 \
  --server-log "$RUN_DIR/server-28155.log" \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 1 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_TRUNCATED_OK_28155 \
  --stop-sequence $'\n```' \
  > "$RUN_DIR/result-28155.json"
status=$?
echo PROBE_EXIT_STATUS=$status
exit 0
````

Result:

```text
status: failed
server_host: 127.0.0.1
server_port: 28155
PROBE_EXIT_STATUS=2
failure_category: chat_needle_stream_expected_answer_not_exact
failure_message: normalized assembled streaming assistant content did not
  exactly equal the expected 256K needle answer
generation_attempted: true
prompt_sent: true
stream_events_received: true
elapsed_seconds: 127.941
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
prompt_chars: 1233760
prompt_unit_chars: 82
max_tokens=1
temperature=0.0
top_p=1.0
seed=0
expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_TRUNCATED_OK_28155
needle_occurrences: 1
needle_position: middle
message_count: 2
message_roles: system,user
match_mode: exact
stop_sequences_configured: true
stop: ["\n```"]
stream: true
n: 1
tokenizer_accounting: transformers.AutoTokenizer fallback encode estimate
prompt_text_recording: false
payload_recording: false
assistant_content_recording: false
```

Streaming endpoint result:

```text
HTTP status: 200
stream_events_received: true
event_count: 2
content_chunk_count: 1
done_seen: true
finish_reason: length
normalized_output_equals_expected: false
normalized_output_length_chars: 1
expected_answer_exact: failed
usage: not_returned
usage_shape: not_returned
stream_parse: passed
stream_content_chunks_received: passed
stream_done_seen: passed
stream_finish_reason_present: passed
```

Review-safe response shape:

```text
choice_count: 1
first_choice_review_safe_keys: delta,finish_reason,index
delta_review_safe_keys: content,role
top_level_review_safe_keys: choices,created,id,model,object
usage_keys: []
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
needle strict exact-comparator failure under the recorded boundary. The
server reached readiness, the streaming request returned HTTP 200, SSE
parsing succeeded, terminal `[DONE]` was received, a final `finish_reason`
was recorded as `length`, and cleanup reported no remaining process-group
PIDs.

The failure is expected and intentionally narrow: `max_tokens=1` was too
small for the expected synthetic answer, so the narrowly normalized assembled
assistant content did not exactly equal
`PTO_CHAT_NEEDLE_256K_STREAM_TRUNCATED_OK_28155`. This is a strict
exact-comparator failure, not a transport/server failure and not
generated-text correctness evidence.

vLLM did not return usage in the streaming response, so usage is recorded as
`not_returned`; this does not weaken the strict exact comparator.

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
