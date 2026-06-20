# vLLM Remote H200 256K Needle Exact Stop-Sequence Probe

This note records a strict remote H200 vLLM synthetic needle exact-output
probe for `deepseek-ai/DeepSeek-V4-Flash` with serving-time stop control
configured. The goal was to test whether an OpenAI-compatible stop sequence
can bound the synthetic answer exactly without weakening the repo-owned
exact-mode comparator.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-256k-needle-exact-stop-probe/`.

## Probe Surface

The repo-owned needle correctness probe accepts repeatable `--stop-sequence`
arguments. When unset, the `/v1/completions` request body omits `stop`. When
set, the request body carries a list-valued `stop` field, matching the
existing vLLM stop-contract request style used elsewhere in this repo.

The exact-mode normalization remains intentionally narrow:

```text
strip leading/trailing whitespace, then strip one surrounding Markdown code
fence when the whole output is fenced
```

It does not remove explanatory sentences, punctuation, unmatched Markdown
fences, or unrelated tokens.

## Resource Plan

The remote H200 host was reachable, and the preserved remote vLLM environment
was present before launch:

```text
.venv-vllm-probe/bin/python: present and executable
.venv-vllm-probe/bin/vllm: present and executable
tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash: present
vllm: 0.23.0
torch: 2.11.0
transformers: 5.12.1
```

The selected physical GPUs for the run were 1 and 7:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
GPU 1: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
GPU 7: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
```

The remote source tree was refreshed with `--sync` from this local worktree
before the run. The sync excludes `.venv-*` and `tmp`, preserving the existing
remote vLLM probe environment and repo-relative model artifact directory.

## Passing Stop-Sequence Exact Diagnostic

Core probe command shape:

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28145 \
  --server-log tmp/vllm-256k-needle-exact-stop-probe/server-28145.log \
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
  --stop-sequence "$(printf %s CmBgYA== | base64 -d)"
```

The final argument decodes to the intended stop sequence `"\n```"`. It was
transported this way only to avoid local shell quoting problems around
Markdown backticks.

Result:

```text
status: passed
server_host: 127.0.0.1
server_port: 28145
PROBE_EXIT_STATUS=0
generation_attempted: true
elapsed_seconds: 130.02
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

Observed response contract:

```text
health_http_200: passed
models_http_200: passed
served_model_listed: passed
max_model_len: passed
completion_http_200: passed
choice_count: passed
choice_fields: passed
model_field: passed
finish_reason: stop
generated_text_length_chars: 33
expected_answer_exact: passed
usage_prompt_tokens_match: passed
usage_completion_bound: passed
usage_total_tokens: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 17
usage.total_tokens: 255816
remaining_process_group_pids: []
```

The exact comparator normalized the short synthetic generated output to
`PTO_NEEDLE_256K_CONTEXT_OK_28143`, exactly matching the expected answer. The
probe did not weaken exact mode to a containment check.

## Interpretation

This is stop-controlled synthetic exact-output evidence for one local-only
remote H200 vLLM request under the recorded two-GPU, near-256K boundary. It
shows that this specific synthetic prompt, expected answer, exact comparator,
and stop sequence produced an exact pass.

It does not establish general generated-text correctness, semantic
correctness, tokenizer semantic correctness, token identity, stop-token
semantic correctness, broad stop-trigger behavior, latency, throughput,
production readiness, broad determinism, or simpler-nv/vLLM integration.

## Review-Safety Omissions

```text
raw prompt text is not recorded
raw request payload is not recorded
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
