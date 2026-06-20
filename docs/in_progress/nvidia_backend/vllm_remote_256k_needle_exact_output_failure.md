# vLLM Remote H200 256K Needle Exact-Output Failure

This note records a strict remote H200 vLLM synthetic needle exact-output
probe for `deepseek-ai/DeepSeek-V4-Flash` with `--max-model-len 262144`.
It follows the prior near-256K needle containment gate and intentionally uses
the stricter `--match-mode exact` contract.

Raw command output is kept under the gitignored repo-relative directory
`tmp/vllm-256k-needle-exact-output-probe/`.

## Probe Surface

The repo-owned needle probe starts `vllm serve`, binds only to `127.0.0.1`,
checks `/health` and `/v1/models`, sends one non-streaming
`/v1/completions` request near a 256K prompt-token budget, validates response
shape, applies the requested match mode, emits structured JSON, terminates the
server process group, and reports remaining process-group PIDs.

Exact-mode normalization is intentionally narrow:

```text
strip leading/trailing whitespace, then strip one surrounding Markdown code
fence when the whole output is fenced
```

It does not remove explanatory sentences, punctuation, unmatched Markdown
fences, or unrelated tokens.

## Resource Plan

The remote source tree was refreshed with `--sync` from the local branch at
base commit `5066ab4d668dd1564e87688e038172931cc55a84` plus this PR's
uncommitted probe changes, preserving the complete ignored artifact directory
and `.venv-vllm-probe`. Remote git metadata was unavailable after sync due
stale worktree metadata, so the synced source tree and structured command
output are the run evidence.

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
GPU 1: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
GPU 7: NVIDIA H200 NVL, compute capability 9.0, driver 580.126.20
```

The fixed local port was:

```text
127.0.0.1:28144
```

## Failing Exact-Output Probe

Core probe command:

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28144 \
  --server-log tmp/vllm-256k-needle-exact-output-probe/server-28144.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143 \
  --match-mode exact
```

Result:

```text
status: failed
server_host: 127.0.0.1
server_port: 28144
PROBE_EXIT_STATUS=2
failure_category: needle_expected_answer_not_exact
failure_message: normalized generated output did not exactly equal the
  expected needle answer
elapsed_seconds: 131.626
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
match_mode: exact
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

Response shape:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
finish_reason: stop
generated_text_length_chars: 37
generated_text_recorded: short_synthetic_output
expected_answer_exact: failed
```

Review-safety omissions:

```text
raw prompt text is not recorded
raw request payload is not recorded
token ID arrays are not recorded
logprob values are not recorded
generated-text digests are not recorded
model artifact contents and symlinks are not recorded
```

Exact short generated output recorded by the synthetic probe:

````text
 PTO_NEEDLE_256K_CONTEXT_OK_28143
```
````

Normalized generated output compared by exact mode:

````text
PTO_NEEDLE_256K_CONTEXT_OK_28143
```
````

The normalized generated output retains the unmatched closing Markdown fence,
so strict exact mode failed. The probe did not weaken the exact-output rule to
make this pass.

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
match_mode=exact
```

The server bound to `127.0.0.1`, returned HTTP 200 from `/health`, returned
the served model with `max_model_len=262144` from `/v1/models`, returned HTTP
200 from one non-streaming `/v1/completions` request, generated short
synthetic output containing the expected answer plus an unmatched closing
Markdown fence, failed the strict exact-output comparison, and shut down with
no remaining process-group PIDs reported by the probe. This is a strict
synthetic exact-output failure record only.

## Non-Claims

- This is not general generated-text correctness evidence.
- This is not semantic correctness evidence.
- This is not throughput or latency evidence.
- This is not production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM integration evidence.
