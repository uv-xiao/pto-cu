# vLLM Remote H200 Chat Exact Truncated Failure Probe

This note records a bounded remote H200 vLLM OpenAI-compatible
chat-completions exact-output failure-mode characterization for
`deepseek-ai/DeepSeek-V4-Flash` with `--max-model-len 262144`. It
intentionally uses `--max-tokens 1`, so the strict exact comparator is
expected to fail while the endpoint still accepts the request and returns a
review-safe response shape.

## Result

```text
status: failed
PROBE_EXIT_STATUS=2
failure_category: chat_canary_expected_answer_not_exact
```

The probe started one local-only vLLM server lifecycle, waited for `/health`
and `/v1/models`, sent one bounded non-streaming `/v1/chat/completions`
request, received HTTP 200, and then failed the strict exact comparator
because the one-token generation budget was insufficient.
This is a strict exact-comparator failure, not a transport/server failure.

## Environment

```text
GPU: 2 selected NVIDIA H200 NVL devices from an 8-GPU host
CUDA_VISIBLE_DEVICES=1,7
python: 3.12.3
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
```

The selected physical GPUs were 1 and 7, exposed to vLLM as exactly two
visible devices.

## Boundary

```text
server_host: 127.0.0.1
server_port: 28150
endpoint: /v1/chat/completions
max_model_len=262144
tensor_parallel_size=2
dtype=bfloat16
quantization=deepseek_v4_fp8
kv_cache_dtype=fp8
gpu_memory_utilization=0.78
distributed_executor_backend=mp
enforce_eager=true
trust_remote_code=false
```

Request controls:

```text
max_tokens=1
temperature=0.0
top_p=1.0
seed=0
stream=false
n=1
message_count: 2
message_roles: system,user
expected_answer: PTO_CHAT_EXACT_CANARY_28149
match_mode: exact
normalization: strip leading/trailing whitespace, then strip one surrounding
  Markdown code fence when the whole output is fenced
stop_sequences_configured: false
```

## Remote Command

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'RUN_DIR=tmp/vllm-chat-exact-truncated-failure-probe
mkdir -p "$RUN_DIR"
set +e
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python timeout --foreground 65m \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_exact_canary_probe.py \
    --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
    --vllm-bin .venv-vllm-probe/bin/vllm \
    --port 28150 \
    --server-log tmp/vllm-chat-exact-truncated-failure-probe/server-28150.log \
    --max-model-len 262144 --tensor-parallel-size 2 \
    --dtype bfloat16 --quantization deepseek_v4_fp8 \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
    --distributed-executor-backend mp --enforce-eager \
    --timeout-seconds 2700 --poll-interval-seconds 10 \
    --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
    --max-tokens 1 --temperature 0.0 --top-p 1.0 \
    --seed 0 --expected-answer PTO_CHAT_EXACT_CANARY_28149 \
  > "$RUN_DIR/result-28150.json"
status=$?
echo PROBE_EXIT_STATUS=$status
exit 0'
```

The remote checkout was refreshed with `--sync` before the run. The preserved
remote model artifacts and `.venv-vllm-probe` environment were used in place.

## Observed Contract

```text
generation_attempted: true
prompt_sent: true
HTTP status: 200
chat_status: failed
finish_reason: length
normalized_output_equals_expected: false
normalized_output_length_chars: 1
expected_answer_exact: failed
usage.prompt_tokens: 33
usage.completion_tokens: 1
usage.total_tokens: 34
usage_completion_bound: passed
usage_total_tokens: passed
```

Review-safe response shape:

```text
choice_count: 1
first_choice_review_safe_keys: finish_reason,index,message
message_review_safe_keys: content,role
top_level_review_safe_keys: choices,created,id,kv_transfer_params,model,object,
  service_tier,system_fingerprint,usage
usage_keys: completion_tokens,prompt_tokens,prompt_tokens_details,total_tokens
```

Cleanup:

```text
remaining_process_group_pids: []
```

Server log evidence:

```text
Using DeepSeek's fp8_ds_mla KV cache format.
Starting vLLM server on http://127.0.0.1:28150
Route: /v1/chat/completions, Methods: POST
GET /health HTTP/1.1" 200 OK
GET /v1/models HTTP/1.1" 200 OK
POST /v1/chat/completions HTTP/1.1" 200 OK
```

## Interpretation

This run shows that the preserved remote H200 vLLM environment accepted one
bounded deterministic OpenAI-compatible `/v1/chat/completions` request for
`deepseek-ai/DeepSeek-V4-Flash`, returned HTTP 200, produced one assistant
choice with a review-safe response shape, and reported bounded usage
accounting. The strict exact comparator then failed as expected because
`max_tokens=1` was deliberately insufficient for the expected canary string.

The result is intentionally narrow. It does not weaken the comparator, does
not change the expected answer, does not fall back to `/v1/completions`, and
does not establish generated-text correctness.

## Review-Safety Omissions

```text
raw prompt text is not recorded
raw request payload is not recorded
raw generated text is not recorded
token ID arrays are not recorded
probability-value details are not recorded
generated-text digests are not recorded
model artifact contents and symlinks are not recorded
private absolute paths are not recorded
```

## Non-Claims

- This is not generated-text correctness evidence.
- This is not semantic correctness evidence.
- This is not tokenizer semantic correctness evidence.
- This is not broad OpenAI chat API coverage.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.
