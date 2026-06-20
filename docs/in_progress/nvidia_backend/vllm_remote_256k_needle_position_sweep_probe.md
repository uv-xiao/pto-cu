# vLLM Remote H200 256K Needle Position Sweep Probe

This note records a strict remote H200 vLLM synthetic needle position sweep
for `deepseek-ai/DeepSeek-V4-Flash` with serving-time stop control configured.
The goal was to check whether one local-only vLLM server lifecycle can satisfy
the existing strict exact comparator when the synthetic needle is placed near
the beginning, middle, and end of a near-256K prompt.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-256k-needle-position-sweep-probe/`.

## Probe Surface

The repo-owned needle correctness probe now accepts `--needle-position` with
`early`, `middle`, and `late` values. The default remains `middle`, preserving
the prior single-request prompt placement. It also accepts
`--needle-position-sweep`, a comma-separated list of unique positions to run
under one server lifecycle.

Sweep summaries are review-safe. They include attempt index, position, status,
HTTP status, finish reason, generated-text length, exact-check result, and
usage token counts. They do not include raw prompt text, raw request payloads,
raw generated text, token ID arrays, logprob values, generated-text digests,
model artifact contents, symlink dumps, or private absolute paths.

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

## Passing Position Sweep

Core probe command shape:

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28148 \
  --server-log tmp/vllm-256k-needle-position-sweep-probe/server-28148.log \
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
  --needle-position-sweep early,middle,late
```

The stop-sequence argument decodes to `"\n```"`. It was transported this way
only to avoid shell quoting problems around Markdown backticks.

Result:

```text
status: passed
server_host: 127.0.0.1
server_port: 28148
PROBE_EXIT_STATUS=0
generation_attempted: true
elapsed_seconds: 190.084
positions_requested: early,middle,late
positions_completed: early,middle,late
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

Per-position prompt accounting:

```text
needle_position: early
filler_units_before_needle: 1598
filler_units_after_needle: 14385

needle_position: middle
filler_units_before_needle: 7991
filler_units_after_needle: 7992

needle_position: late
filler_units_before_needle: 14384
filler_units_after_needle: 1599
```

Sweep attempt summaries:

```text
attempt_index: 1
needle_position: early
status: passed
http_status: 200
finish_reason: stop
generated_text_length_chars: 33
exact_check: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 17
usage.total_tokens: 255816

attempt_index: 2
needle_position: middle
status: passed
http_status: 200
finish_reason: stop
generated_text_length_chars: 33
exact_check: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 17
usage.total_tokens: 255816

attempt_index: 3
needle_position: late
status: passed
http_status: 200
finish_reason: stop
generated_text_length_chars: 33
exact_check: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 16
usage.total_tokens: 255815
```

Cleanup:

```text
remaining_process_group_pids: []
```

## Interpretation

This is stop-controlled synthetic exact-output position coverage for three
local-only remote H200 vLLM requests under one recorded two-GPU, near-256K
server lifecycle. It shows that this specific synthetic prompt class,
expected answer, exact comparator, stop sequence, and position set produced
exact passes for early, middle, and late needle placement.

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
