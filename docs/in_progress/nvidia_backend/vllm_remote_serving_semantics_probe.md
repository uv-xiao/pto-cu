# vLLM Remote H200 Serving-Semantics Probe

This note records a bounded remote H200 serving-semantics probe for
`deepseek-ai/DeepSeek-V4-Flash`. It reuses the complete repo-relative
artifact directory, the local-only vLLM server boundary, the existing
structural response-contract validation, and the selected JIT-warning
server-log windows from prior gates. It then sends two identical bounded
deterministic completion requests and compares response observations at the
API boundary.

Raw command output is kept under the gitignored local directory
`tmp/vllm-serving-semantics-probe/`.

## Probe Surface

A repo-owned serving-semantics probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends two identical bounded
non-streaming `/v1/completions` requests with explicit deterministic sampler
settings, validates the structural response contract for both responses,
compares selected response observations, counts selected Triton JIT warning
strings by server-log request windows, terminates the server process group,
and reports remaining process-group PIDs.

The request payload for both completion requests is:

```json
{
  "max_tokens": 8,
  "model": "deepseek-ai/DeepSeek-V4-Flash",
  "n": 1,
  "prompt": "Hello. Keep this deterministic serving probe bounded.",
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
HTTP 200 from first deterministic /v1/completions
HTTP 200 from repeat deterministic /v1/completions
exactly one response choice for both completion responses
choice text and finish_reason fields present
response model field present
usage prompt/completion/total token fields present
usage.completion_tokens within request max_tokens
usage.total_tokens >= usage.prompt_tokens
token_ids length matches usage.completion_tokens when list-valued token_ids exist
server log inspected for selected JIT warning strings by request window
server process group cleanup leaves no remaining PIDs
```

Serving-semantics checks:

```text
deterministic repeat returns the same completion text digest
deterministic repeat returns the same completion text length
deterministic repeat returns the same finish_reason
deterministic repeat returns the same usage accounting
generated text is not recorded or checked for correctness
```

The selected server-log warning pattern for this gate is:

```text
Triton kernel JIT compilation during inference
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the resource-plan
capture and before the serving-semantics run. The ignored repo-relative
artifact directory and checkout-local `.venv-vllm-probe` were preserved. The
synced remote checkout did not expose usable Git metadata, so the run relies
on the `--sync` command as the source-tree refresh evidence.

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
transformers: 5.12.1
```

The selected physical GPUs were 1 and 7, exposed to vLLM as exactly two
visible devices:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
```

They were selected because this pair passed the prior bounded model-load,
server-health, inference-smoke, response-contract, warmup-shape, and
request-shape variation gates, and the fresh memory check still showed
enough free memory for the 0.78 utilization boundary. This selection is not
performance evidence.

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28128 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28128/health
GET http://127.0.0.1:28128/v1/models
POST http://127.0.0.1:28128/v1/completions  # first deterministic
POST http://127.0.0.1:28128/v1/completions  # repeat deterministic
```

Timeouts:

```text
outer timeout=65m
readiness timeout=2700s
request timeout=180s
log settle after each request=2s
```

## Passing Serving-Semantics Probe

Exact command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set +euo pipefail
RUN_LOG=tmp/vllm-serving-semantics-probe/server-28128.log
mkdir -p tmp/vllm-serving-semantics-probe
printf "== pre-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== serving semantics json ==\n"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_serving_semantics_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28128 \
  --server-log "$RUN_LOG" \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --max-tokens 8 --temperature 0.0 --top-p 1.0 \
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
tail -n 380 "$RUN_LOG" 2>/dev/null
exit "$rc"'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 125.057
server_host: 127.0.0.1
server_port: 28128
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28128 \
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
/health: HTTP 200 after 12 polling attempts
/v1/models: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 4096
first deterministic /v1/completions: HTTP 200
repeat deterministic /v1/completions: HTTP 200
```

First response-contract result:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
usage.prompt_tokens: 9
usage.completion_tokens: 8
usage.total_tokens: 17
request max_tokens: 8
finish_reason: length
stop_reason: null
text_length_chars: 34
token_ids check: not_present
```

Repeat response-contract result:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
usage.prompt_tokens: 9
usage.completion_tokens: 8
usage.total_tokens: 17
request max_tokens: 8
finish_reason: length
stop_reason: null
text_length_chars: 34
token_ids check: not_present
```

Serving-semantics result:

```text
text digest match: passed
text length match: passed
finish_reason match: passed
usage accounting match: passed
text_sha256: 7fc094e1191328a70f430d0b9839ae781466ccaee00799e90f99f81dacb2772c
```

The raw generated text was not recorded and was not compared against an
expected answer. The digest is used only to compare the two responses from
this exact deterministic request pair.

Recorded response shape for both completions:

```text
top-level keys: choices, created, id, kv_transfer_params, model, object,
  service_tier, system_fingerprint, usage
first choice keys: finish_reason, index, logprobs, prompt_logprobs,
  prompt_token_ids, routed_experts, stop_reason, text, token_ids
usage keys: completion_tokens, prompt_tokens, prompt_tokens_details,
  total_tokens
```

## JIT Warning Evidence

The server log was inspected with byte offsets captured after readiness and
after each completion request plus a 2-second log-settle interval. This is
practical request-window attribution for the selected log substring, not a
general kernel-compilation proof.

Observed counts for the selected warning pattern:

```text
before_first: 0
first_request: 11
repeat_request: 0
full_before_cleanup: 11
```

The first-request warning lines matched the same selected pattern and named
these kernels:

```text
_compute_slot_mapping_kernel
_compressed_slot_mapping_kernel
_build_prefill_chunk_metadata_kernel
_build_c128a_topk_metadata_kernel
_compute_prefill_metadata_kernel
_combine_topk_swa_indices_kernel
_fused_inv_rope_fp8_quant_per_head
_save_partial_states_kernel
_fused_kv_compress_norm_rope_insert_indexer_attn
_compute_swa_indices_and_lens_kernel
_compute_global_topk_indices_and_lens_kernel
```

The repeat deterministic request had zero fresh lines matching
`Triton kernel JIT compilation during inference` in its request window.

## Shutdown Behavior

The probe terminated the server process group after the repeat request and
serving-semantics comparison:

```text
terminated: true
killed: false
returncode_after_cleanup: 0
remaining_process_group_pids: []
cleanup.status: passed
```

The server log recorded vLLM shutdown and API server exit. It also emitted a
process-manager force-kill warning, plus Python `resource_tracker` semaphore
and shared-memory cleanup warnings at shutdown. The probe still reported no
remaining process-group PIDs, and the immediate post-run selected-GPU memory
snapshot returned to the pre-run baseline:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

## Interpretation

Two identical bounded deterministic local-only completion requests passed the
same structural response contract and returned matching API-boundary
observations: text digest, text length, `finish_reason`, and usage
accounting. This is the exact two-H200 boundary:

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
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
- This does not commit raw model artifacts, venvs, command dumps, server
  logs, or `tmp/` symlinks.
