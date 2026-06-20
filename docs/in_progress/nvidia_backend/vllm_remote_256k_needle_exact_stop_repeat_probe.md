# vLLM Remote H200 256K Needle Exact Stop Repeat Probe

This note records a strict remote H200 vLLM synthetic needle exact-output
repeat probe for `deepseek-ai/DeepSeek-V4-Flash` with serving-time stop
control configured. The goal was to check whether one local-only vLLM server
lifecycle can satisfy the existing strict exact comparator for a small number
of repeated near-boundary requests.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-256k-needle-exact-stop-repeat-probe/`.

## Probe Surface

The repo-owned needle correctness probe now accepts `--repeat-count`, which
defaults to `1`. The default path preserves the previous single-request result
shape. When `--repeat-count` is greater than `1`, the probe starts one server
lifecycle, sends the same bounded `/v1/completions` request repeatedly, runs
the existing response-contract and exact comparator on each response, and
fails the aggregate result if any attempt fails.

Repeat summaries are review-safe. They include attempt index, status, HTTP
status, finish reason, generated-text length, exact-check result, and usage
token counts. They do not include raw prompt text, raw request payloads, raw
generated text, token ID arrays, logprob values, generated-text digests, model
artifact contents, symlink dumps, or private absolute paths.

## Remote Preflight

The prior successful remote environment path was checked before launch:

```text
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact
.venv-vllm-probe/bin/python: present and executable
.venv-vllm-probe/bin/vllm: present and executable
tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash: present
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
transformers: 5.12.1
```

The selected physical GPUs for the run were 1 and 7:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
GPU 1: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
GPU 7: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
```

The remote source tree was refreshed with `--sync` from this worker worktree
before the run. The sync excludes `.venv-*` and `tmp`, preserving the existing
remote vLLM probe environment and repo-relative model artifact directory.

## Passing Repeat Diagnostic

Core probe command shape:

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28146 \
  --server-log tmp/vllm-256k-needle-exact-stop-repeat-probe/server-28146.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143 \
  --match-mode exact \
  --stop-sequence "$(printf %s CmBgYA== | base64 -d)" \
  --repeat-count 3
```

The stop-sequence argument decodes to `"\n```"`. It was transported this way
only to avoid shell quoting problems around Markdown backticks.

Result:

```text
status: passed
server_host: 127.0.0.1
server_port: 28146
PROBE_EXIT_STATUS=0
generation_attempted: true
elapsed_seconds: 172.092
repeat_count: 3
passed_attempts: 3
failed_attempts: 0
```

Request controls:

```text
max_model_len=262144
tensor_parallel_size=2
target_prompt_tokens=255800
actual_prompt_tokens=255799
prompt_chars: 1230965
max_tokens=64
expected_answer: PTO_NEEDLE_256K_CONTEXT_OK_28143
needle_occurrences: 1
match_mode: exact
stop_sequences_configured: true
stop: ["\n```"]
stream: false
echo: false
logprobs: false
```

Repeat attempt summaries:

```text
attempt_index: 1
status: passed
http_status: 200
finish_reason: stop
generated_text_length_chars: 33
exact_check: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 17
usage.total_tokens: 255816

attempt_index: 2
status: passed
http_status: 200
finish_reason: stop
generated_text_length_chars: 33
exact_check: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 17
usage.total_tokens: 255816

attempt_index: 3
status: passed
http_status: 200
finish_reason: stop
generated_text_length_chars: 33
exact_check: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 17
usage.total_tokens: 255816
```

Cleanup:

```text
remaining_process_group_pids: []
```

## Interpretation

This is stop-controlled synthetic exact-output repeat evidence for three
local-only remote H200 vLLM requests under one recorded two-GPU, near-256K
server lifecycle. It shows that this specific synthetic prompt, expected
answer, exact comparator, stop sequence, and repeat count produced three
exact passes.

It does not establish general generated-text correctness, semantic
correctness, tokenizer semantic correctness, token identity, stop-token
semantic correctness, broad stop-trigger behavior, broad determinism,
latency, throughput, production readiness, or simpler-nv/vLLM integration.

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
