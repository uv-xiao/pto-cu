# vLLM Remote H200 Request-Shape Variation Probe

This note records a bounded remote H200 request-shape variation probe for
`deepseek-ai/DeepSeek-V4-Flash`. It reuses the complete repo-relative
artifact directory, the local-only vLLM server boundary, and the known
warmup request shape from the prior warmup-shape gate. It then sends a
separate bounded completion request with a deliberately different prompt and
token shape.

Raw command output is kept under the gitignored local directory
`tmp/vllm-request-shape-variation-probe/`.

## Probe Surface

A repo-owned request-shape variation probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends one labeled warmup
`/v1/completions` request, sends one same-shape follow-up request to preserve
the prior baseline inside this run, sends one labeled variation
`/v1/completions` request, validates the structural response contract for
each completion response, counts selected Triton JIT warning strings by
server-log byte windows, terminates the server process group, and reports
remaining process-group PIDs.

The warmup and same-shape follow-up payloads were:

```json
{
  "max_tokens": 4,
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "n": 1,
  "prompt": "Hello",
  "seed": 0,
  "stream": false,
  "temperature": 0.0,
  "top_p": 1.0
}
```

The variation payload was:

```json
{
  "max_tokens": 8,
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "n": 1,
  "prompt": "Hello. Provide one short sentence about bounded CUDA serving probes.",
  "seed": 0,
  "stream": false,
  "temperature": 0.0,
  "top_p": 1.0
}
```

Contract checks:

```text
HTTP 200 from /health
HTTP 200 from /v1/models
HTTP 200 from warmup /v1/completions
HTTP 200 from same-shape follow-up /v1/completions
HTTP 200 from variation /v1/completions
exactly one response choice for every completion response
choice text and finish_reason fields present
response model field present
usage prompt/completion/total token fields present
usage.completion_tokens within request max_tokens
usage.total_tokens >= usage.prompt_tokens
token_ids length matches usage.completion_tokens when list-valued token_ids exist
server log inspected for selected JIT warning strings by request window
server process group cleanup leaves no remaining PIDs
```

The selected server-log warning pattern for this gate was:

```text
Triton kernel JIT compilation during inference
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the resource-plan
capture and before the request-shape variation run. The ignored
repo-relative artifact directory and checkout-local `.venv-vllm-probe` were
preserved. The local branch was `vllm-request-shape-variation-probe`, based
on commit `cd77af73a920a47ec5b54dc7419c9f135855b98a`. The synced remote
checkout did not have usable Git metadata, so the run relies on the `--sync`
command as the source-tree refresh evidence.

Remote tooling and package versions:

```text
GPU: 8 x NVIDIA H200 NVL
driver: 580.126.20
CUDA toolkit: /usr/local/cuda, Build cuda_12.8.r12.8/compiler.35404655_0
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

They were selected because this pair passed the prior bounded model-load,
server-health, inference-smoke, response-contract, and warmup-shape gates,
and the fresh memory check still showed enough free memory for the 0.78
utilization boundary. This selection is not performance evidence.

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28127 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28127/health
GET http://127.0.0.1:28127/v1/models
POST http://127.0.0.1:28127/v1/completions  # warmup
POST http://127.0.0.1:28127/v1/completions  # follow-up same shape
POST http://127.0.0.1:28127/v1/completions  # variation
```

Timeouts:

```text
outer timeout: 65m
readiness timeout: 2700s
request timeout: 180s
cleanup timeout: 60s
log settle after each request: 2s
```

## Passing Request-Shape Variation Probe

Exact command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set +euo pipefail
RUN_LOG=tmp/vllm-request-shape-variation-probe/server-28127.log
mkdir -p tmp/vllm-request-shape-variation-probe
printf "== pre-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== request shape variation json ==\n"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_request_shape_variation_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28127 \
  --server-log "$RUN_LOG" \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --variation-max-tokens 8 --log-settle-seconds 2
rc=$?
printf "== probe exit code ==\n%s\n" "$rc"
printf "== post-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== server log jit warnings ==\n"
grep -n "Triton kernel JIT compilation during inference" "$RUN_LOG" \
  2>/dev/null || true
printf "== server log tail ==\n"
tail -n 360 "$RUN_LOG" 2>/dev/null
exit "$rc"'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 163.042
server_host: 127.0.0.1
server_port: 28127
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28127 \
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
/health: HTTP 200 after 13 polling attempts
/v1/models: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 4096
warmup /v1/completions: HTTP 200
follow-up same-shape /v1/completions: HTTP 200
variation /v1/completions: HTTP 200
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

Variation response-contract result:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
usage.prompt_tokens: 13
usage.completion_tokens: 8
usage.total_tokens: 21
request max_tokens: 8
token_ids check: not_present
```

All three responses included the same structural response-shape keys recorded
by the response-contract gate. The probe did not compare or record generated
text as correctness evidence.

## JIT Warning Evidence

The server log was inspected with byte offsets captured after readiness and
after each completion request plus a 2-second log-settle interval. This is
practical request-window attribution for the selected log substring, not a
general kernel-compilation proof.

Observed counts for the selected warning pattern:

```text
before_warmup: 0
warmup_request: 8
followup_same_shape_request: 0
variation_request: 3
full_before_cleanup: 11
```

The eight warmup-window warning lines matched the same selected pattern and
named these kernels:

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

The same-shape follow-up request had zero fresh lines matching
`Triton kernel JIT compilation during inference` in its request window.

The variation request had three fresh lines matching the selected pattern in
its request window:

```text
_build_prefill_chunk_metadata_kernel
_compute_prefill_metadata_kernel
_combine_topk_swa_indices_kernel
```

This is only an observation for this exact three-request command, selected
warning pattern, and request shapes; it is not a broad claim that warmup
eliminates all JIT behavior or that any request shape is semantically correct.

## Shutdown Behavior

The probe terminated the server process group after the variation request
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

After the known warmup shape ran, a separate bounded completion request with a
different prompt/token shape passed the same structural response contract.
The selected Triton JIT warning pattern appeared 3 times in the variation
request log window, while the intervening same-shape follow-up request had
zero fresh selected warning lines.

This is the exact two-H200 boundary:

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

The next reviewable gate can move to a bounded serving-semantics contract that
still avoids generated-text correctness claims, or can deliberately exercise a
second explicit bounded variation shape. Keep that gate separate from
throughput, latency, long-context, and production-readiness claims.
