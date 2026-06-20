# vLLM Remote H200 256K Needle Exact Truncated Failure Probe

This note records an expected failure-mode characterization for the strict
remote H200 vLLM synthetic needle exact-output gate for
`deepseek-ai/DeepSeek-V4-Flash`. The run kept the same near-256K prompt
budget and stop-controlled exact comparator as the previous passing gate, but
intentionally set `max_tokens=1`, which is too small to emit the full expected
synthetic answer.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-256k-needle-exact-truncated-failure-probe/`.

## Probe Surface

The repo-owned needle correctness probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends one non-streaming
`/v1/completions` request near a 256K prompt-token budget, validates response
shape, applies the requested match mode, emits structured JSON, terminates the
server process group, and reports remaining process-group PIDs.

This run is intentionally expected to fail strict exact correctness. The
expected evidence is a clean serving/request path followed by
`needle_expected_answer_not_exact`, not a false pass and not a setup failure.

Exact-mode normalization remains intentionally narrow:

```text
strip leading/trailing whitespace, then strip one surrounding Markdown code
fence when the whole output is fenced
```

It does not remove unrelated tokens or weaken exact mode to containment.

## Remote Preflight

The prior successful remote environment path was checked before launch:

```text
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact
.venv-vllm-probe/bin/python: present and executable
.venv-vllm-probe/bin/vllm: present and executable
tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash: present
examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py: present
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
python: 3.12.3
```

Remote git metadata was unavailable due stale worktree metadata, so the
preflight treated the preserved vLLM environment, model artifacts, repo-owned
script, and structured command output as the run evidence. This was not a
setup failure because the required venv, vLLM executable, and artifact path
were present.

The selected physical GPUs for the run were 1 and 7:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
GPU 1: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
GPU 7: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
```

## Truncated Exact Failure Diagnostic

Core probe command shape:

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28147 \
  --server-log tmp/vllm-256k-needle-exact-truncated-failure-probe/server-28147.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 1 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143 \
  --match-mode exact \
  --stop-sequence "$(printf %s CmBgYA== | base64 -d)"
```

The stop-sequence argument decodes to `"\n```"`. It was transported this way
only to avoid shell quoting problems around Markdown backticks.

The outer wrapper preserved the probe exit status and exited 0 so this
expected comparator failure could be documented:

```text
status: failed
server_host: 127.0.0.1
server_port: 28147
PROBE_EXIT_STATUS=2
failure_category: needle_expected_answer_not_exact
failure_message: normalized generated output did not exactly equal the
  expected needle answer
generation_attempted: true
elapsed_seconds: 159.605
```

Request controls:

```text
max_model_len=262144
tensor_parallel_size=2
target_prompt_tokens=255800
actual_prompt_tokens=255799
prompt_chars: 1230965
max_tokens=1
expected_answer: PTO_NEEDLE_256K_CONTEXT_OK_28143
needle_occurrences: 1
match_mode: exact
stop_sequences_configured: true
stop: ["\n```"]
stream: false
echo: false
logprobs: false
temperature: 0.0
top_p: 1.0
seed: 0
prompt_text_recorded: false
payload_recorded: false
tokenizer_accounting: transformers.AutoTokenizer local encode
```

Endpoint and response contract:

```text
health_http_200: passed
models_http_200: passed
served_model_listed: passed
max_model_len: passed
completion_http_200: passed
choice_count: passed
choice_fields: passed
model_field: passed
finish_reason: length
generated_text_length_chars: 2
normalized_generated_text: P
expected_answer_exact: failed
usage_prompt_tokens_match: passed
usage_completion_bound: passed
usage_total_tokens: passed
usage.prompt_tokens: 255799
usage.completion_tokens: 1
usage.total_tokens: 255800
remaining_process_group_pids: []
```

The server accepted the request and returned HTTP 200 with one response
choice. Generation was attempted, but a one-token completion budget produced
only the first synthetic output fragment after normalization, so strict exact
mode correctly failed instead of reporting a false pass.

## Interpretation

This is expected failure-mode characterization for one local-only remote H200
vLLM request under the recorded two-GPU, near-256K boundary. It proves that
the same evidence path can preserve a clean request path and still record an
expected exact comparator failure when the completion budget is insufficient.

It is not serving correctness evidence and does not weaken exact mode to
containment.

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
- This is not stop-trigger or stop-token semantic correctness evidence.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.
