# vLLM Remote H200 Chat 256K Needle Streaming Usage Contract Probe

This note records one bounded OpenAI-compatible streaming
chat-completions synthetic needle usage-contract attempt for
`deepseek-ai/DeepSeek-V4-Flash`. The run used
`stream_options.include_usage=true` and completed with a review-safe failure:
usage was returned in a final usage-bearing streaming event, but the shared
streaming parser rejected that event because it contained zero choices.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-chat-256k-needle-stream-usage-contract-probe/`.

## Probe Surface

The repo-owned usage-contract entry point reuses the chat 256K streaming
needle prompt builder, lifecycle, readiness contract, and exact
normalization, then adds the OpenAI streaming usage option:

````text
endpoint: /v1/chat/completions
stream: true
stream_options.include_usage=true
expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28157
stop: ["\n```"]
match_mode: exact
````

The intended contract requires returned streaming usage and strict exact
output matching. This run failed before the exact comparator because the
streaming parser treated the final usage-only event shape as invalid.

## Remote Preflight

The preserved remote vLLM probe environment was reused. The selected remote
checkout was refreshed with `--sync`, which excludes `.venv`, `.venv-*`,
`build`, `tmp`, Python caches, and pytest caches, preserving the existing
remote vLLM probe venv and repo-relative model artifact directory.

The preflight showed H200 visibility and required prerequisites:

```text
8 x NVIDIA H200 NVL
compute capability: 9.0
driver: 580.126.20
memory.total: 143771 MiB per GPU
python_status=0
vllm_status=0
artifact_status=0
```

The preserved checkout reported stale git worktree metadata during
`git rev-parse HEAD`; that value is not used as evidence. The source tree was
then refreshed from this worker with `--sync` before the bounded probe.

## Usage-Contract Attempt

Core probe command shape:

````bash
.agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh \
  --remote-dir <remote-pto-cu> --sync -- \
  bash -lc '<bounded command that preserves PROBE_EXIT_STATUS>'

CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_usage_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28157 \
  --server-log tmp/vllm-chat-256k-needle-stream-usage-contract-probe/server-28157.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28157
````

Result:

```text
status: failed
server_host: 127.0.0.1
server_port: 28157
PROBE_EXIT_STATUS=2
failure_category: chat_needle_stream_choice_shape
failure_message: streaming chunk must contain exactly one choice
generation_attempted: true
prompt_sent: true
stream_events_received: true
elapsed_seconds: 130.449
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
prompt_chars: 1233756
prompt_unit_chars: 82
max_tokens=64
temperature=0.0
top_p=1.0
seed=0
expected_answer: PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28157
needle_occurrences: 1
needle_position: middle
message_count: 2
message_roles: system,user
match_mode: exact
stop_sequences_configured: true
stop: ["\n```"]
stream: true
stream_options.include_usage=true
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
event_count: 22
content_chunk_count: 18
done_seen: false
finish_reason: null
normalized_output_equals_expected: not_evaluated
expected_answer_exact: not_evaluated
usage_presence: returned
usage_accounting_checks: not_evaluated_after_stream_parse_failure
```

Review-safe response shape observations:

```text
ordinary streaming content events:
  choice_count: 1
  first_choice_review_safe_keys: delta,finish_reason,index
  delta_review_safe_keys: content or content,role
  top_level_review_safe_keys: choices,created,id,model,object
  usage_keys: []
final usage-bearing event:
  choice_count: 0
  top_level_review_safe_keys: choices,created,id,model,object,system_fingerprint,usage
  usage_keys: completion_tokens,prompt_tokens,total_tokens
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
needle usage-contract attempt under the recorded 262144-token vLLM server
boundary. The server reached readiness, the streaming request returned HTTP
200, content SSE events were parsed, and vLLM returned a streaming usage
object with prompt, completion, and total token keys.

The probe did not pass. Its failure category was
`chat_needle_stream_choice_shape`: the final usage-bearing event had
`choice_count: 0`, while the shared streaming parser required every JSON
streaming event to contain exactly one choice. Because parsing failed at that
event, terminal `[DONE]`, final `finish_reason`, strict exact output
matching, and usage-accounting pass/fail checks are not established by this
run. Exact output may have passed only if a future structured result records
the exact comparator as passed.

This failure is review-useful usage-contract evidence, not a transport/server
failure. It shows that `stream_options.include_usage=true` can return usage
keys on this boundary, while the current gate still fails before it can claim
usage accounting or exact-output success.

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
