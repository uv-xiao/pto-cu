# vLLM Remote H200 Warmup-Shape Probe

This note records a bounded remote H200 warmup-shape probe for
`deepseek-ai/DeepSeek-V4-Flash`. It reuses the complete repo-relative
artifact directory and the local-only vLLM server boundary from prior gates,
then sends one labeled warmup completion request and one follow-up same-shape
completion request. It validates only the existing structural completion
response contract and inspects the server log for selected Triton JIT warning
strings.

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-warmup-shape-probe/`.

## Probe Surface

A repo-owned warmup-shape probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends two bounded
non-streaming completion requests with the same shape, validates the response
contract for both responses, records selected JIT warning counts by server-log
byte windows, terminates the server process group, and reports remaining
process-group PIDs.

The warmup request and follow-up request use the same payload:

```json
{
  "endpoint": "/v1/completions",
  "payload": {
    "max_tokens": 4,
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "n": 1,
    "prompt": "Hello",
    "seed": 0,
    "stream": false,
    "temperature": 0.0,
    "top_p": 1.0
  },
  "limits": {
    "max_tokens": 4,
    "n": 1,
    "prompt_chars": 5,
    "seed": 0,
    "stream": false,
    "temperature": 0.0,
    "top_p": 1.0
  }
}
```

Contract checks:

```text
HTTP 200 from /health
HTTP 200 from /v1/models
HTTP 200 from warmup /v1/completions
HTTP 200 from follow-up same-shape /v1/completions
exactly one response choice for each completion response
choice text and finish_reason fields present
response model field present
usage prompt/completion/total token fields present
usage.completion_tokens within request max_tokens
usage.total_tokens >= usage.prompt_tokens
token_ids length matches usage.completion_tokens when list-valued token_ids exist
server log inspected for selected JIT warning strings
server process group cleanup leaves no remaining PIDs
```

The selected server-log warning pattern for this gate was:

```text
Triton kernel JIT compilation during inference
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the resource-plan
capture and before the warmup-shape run. The ignored repo-relative artifact
directory and checkout-local `.venv-vllm-probe` were preserved. The branch was
synced from local branch `vllm-warmup-shape-probe` based on commit
`b50e76a7072ef9dcadfc58207cae78834d189316`.

Remote tooling and package versions:

```text
GPU: 8 x NVIDIA H200 NVL
driver: 580.126.20
CUDA toolkit: /usr/local/cuda, nvcc 12.8 V12.8.61
python: 3.12.3
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
triton: 3.6.0
```

The selected physical GPUs were 1 and 7, exposed to vLLM as exactly two
visible devices:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
```

They were selected because this pair previously passed the bounded model-load,
server-health, inference-smoke, and response-contract gates, and the fresh
memory check still showed enough free memory for the 0.78 utilization
boundary. This selection is not performance evidence.

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28126 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28126/health
GET http://127.0.0.1:28126/v1/models
POST http://127.0.0.1:28126/v1/completions  # warmup
POST http://127.0.0.1:28126/v1/completions  # follow-up same shape
```

Timeouts:

```text
outer timeout: 65m
readiness timeout: 2700s
request timeout: 180s
cleanup timeout: 60s
log settle after each request: 2s
```

## Passing Warmup-Shape Probe

Exact command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set +euo pipefail
RUN_LOG=tmp/vllm-warmup-shape-probe/server-28126.log
mkdir -p tmp/vllm-warmup-shape-probe
printf "== pre-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== warmup shape json ==\n"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_warmup_shape_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28126 \
  --server-log "$RUN_LOG" \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --log-settle-seconds 2
rc=$?
printf "== probe exit code ==\n%s\n" "$rc"
printf "== post-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== server log jit warnings ==\n"
grep -n "Triton kernel JIT compilation during inference" "$RUN_LOG" \
  2>/dev/null || true
printf "== server log tail ==\n"
tail -n 320 "$RUN_LOG" 2>/dev/null
exit "$rc"'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 115.046
server_host: 127.0.0.1
server_port: 28126
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28126 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer-mode deepseek_v4 \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

Endpoint results:

```text
/health: HTTP 200 after 11 polling attempts
/v1/models: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 4096
warmup /v1/completions: HTTP 200
follow-up same-shape /v1/completions: HTTP 200
```

Warmup response-contract result:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
usage.prompt_tokens: 1
usage.completion_tokens: 4
usage.total_tokens: 5
request max_tokens: 4
token_ids check: not_present
```

Follow-up same-shape response-contract result:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
usage.prompt_tokens: 1
usage.completion_tokens: 4
usage.total_tokens: 5
request max_tokens: 4
token_ids check: not_present
```

Both responses included the same structural response-shape keys recorded by
the response-contract gate. The probe did not compare or record generated text
as correctness evidence.

## JIT Warning Evidence

The server log was inspected with byte offsets captured after readiness,
after the warmup request plus a 2-second log-settle interval, and after the
follow-up same-shape request plus another 2-second log-settle interval. This
is practical request-window attribution for the selected log substring, not a
general kernel-compilation proof.

Observed counts for the selected warning pattern:

```text
before_warmup: 0
warmup_request: 8
followup_same_shape_request: 0
full_before_cleanup: 8
```

The eight warmup-window warning lines named these kernels:

```text
_compute_slot_mapping_kernel
_compressed_slot_mapping_kernel
_build_c128a_topk_metadata_kernel
_compute_swa_indices_and_lens_kernel
_fused_inv_rope_fp8_quant_per_head
_save_partial_states_kernel
_fused_kv_compress_norm_rope_insert_indexer_attn
_compute_global_topk_indices_and_lens_kernel
```

The follow-up same-shape request had zero fresh lines matching
`Triton kernel JIT compilation during inference` in its request window. This
is only an observation for this exact two-request command, selected warning
pattern, and request shape; it is not a broad claim that warmup eliminates all
JIT behavior.

## Shutdown Behavior

The probe terminated the server process group after the follow-up same-shape
contract validation:

```text
terminated: true
killed: false
returncode_after_cleanup: 0
remaining_process_group_pids: []
cleanup.status: passed
```

The server log recorded vLLM shutdown and API server exit. It also emitted a
worker shutdown warning and a Python `resource_tracker` shared-memory cleanup
warning at shutdown. The probe still reported no remaining process-group PIDs,
and the immediate post-run selected-GPU memory snapshot returned to the
pre-run baseline:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

## Interpretation

The remote H200 vLLM environment can start the local-only server, pass
`/health` and `/v1/models`, return HTTP 200 from one labeled warmup
`/v1/completions` request, return HTTP 200 from one follow-up same-shape
`/v1/completions` request, satisfy the response-contract checks for both
responses, inspect the selected Triton JIT warning string by request window,
and shut down with no remaining process-group PIDs under this explicit
two-H200 boundary:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
max_model_len=4096
dtype=bfloat16
quantization=deepseek_v4_fp8
kv_cache_dtype=fp8
gpu_memory_utilization=0.78
enforce_eager=true
distributed_executor_backend=mp
outer timeout=65m
readiness timeout=2700s
request timeout=180s
```

## Non-Claims

- This is not generated-text correctness evidence.
- This is not tokenizer semantic correctness or prompt correctness evidence.
- This is not 256K context behavior evidence.
- This is not latency, throughput, or production-readiness evidence.
- This is not a broad claim that warmup eliminates all JIT behavior.
- This is not simpler-nv or vLLM kernel integration evidence.
- This did not commit raw model artifacts, venvs, command dumps, server logs,
  or `tmp/` symlinks.

## Next Gate

The next reviewable gate can choose a separate serving-semantics check, or a
separate explicitly bounded command that varies the request shape and records
whether new JIT warning lines appear. Keep that gate separate from generated
text correctness, throughput, latency, long-context, and production-readiness
claims.
