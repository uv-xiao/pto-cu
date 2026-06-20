# vLLM Remote H200 Response-Contract Probe

This note records a bounded remote H200 OpenAI-compatible completion
response-contract probe for `deepseek-ai/DeepSeek-V4-Flash`. It reuses the
complete repo-relative artifact directory and the local-only vLLM server
boundary from prior gates, then validates structural response invariants for
one `/v1/completions` request. It does not validate generated text.

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-response-contract-probe/`.

## Probe Surface

A repo-owned response-contract probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends one bounded non-streaming
completion request, validates the response structure, terminates the server
process group, and reports remaining process-group PIDs.

Default request for this gate:

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
HTTP 200 from /v1/completions
exactly one response choice
choice text and finish_reason fields present
response model field present
usage prompt/completion/total token fields present
usage.completion_tokens within request max_tokens
usage.total_tokens >= usage.prompt_tokens
token_ids length matches usage.completion_tokens when list-valued token_ids exist
local-only non-streaming bounded request with explicit sampler settings
server process group cleanup leaves no remaining PIDs
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the resource-plan
capture and before the response-contract run. The ignored repo-relative
artifact directory and checkout-local `.venv-vllm-probe` were preserved.

Remote tooling and package versions:

```text
GPU: 8 x NVIDIA H200 NVL
driver: 580.126.20
CUDA toolkit: /usr/local/cuda, nvcc 12.8 V12.8.61
python: 3.12.3
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
```

The selected physical GPUs were 1 and 7, exposed to vLLM as exactly two
visible devices:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
```

The topology report showed GPU 1 and GPU 7 connected through `SYS`, not
NVLink. They were selected because this pair previously passed the bounded
model-load, server-health, and inference-smoke gates, and the fresh memory
check still showed enough free memory for the 0.78 utilization boundary. This
selection is not performance evidence.

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28125 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28125/health
GET http://127.0.0.1:28125/v1/models
POST http://127.0.0.1:28125/v1/completions
```

Timeouts:

```text
outer timeout: 60m
readiness timeout: 2700s
request timeout: 180s
cleanup timeout: 60s
```

## Passing Response-Contract Probe

Exact command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set +euo pipefail
RUN_LOG=tmp/vllm-response-contract-probe/server-28125.log
mkdir -p tmp/vllm-response-contract-probe
printf "== pre-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== response contract json ==\n"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 60m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28125 \
  --server-log "$RUN_LOG" \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 --seed 0
rc=$?
printf "== probe exit code ==\n%s\n" "$rc"
printf "== post-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== server log tail ==\n"
tail -n 280 "$RUN_LOG" 2>/dev/null
exit "$rc"'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 109.204
server_host: 127.0.0.1
server_port: 28125
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28125 \
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
/v1/completions: HTTP 200
```

Response-contract result:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
usage.prompt_tokens: 1
usage.completion_tokens: 4
usage.total_tokens: 5
request max_tokens: 4
token_ids check: not_present
```

The first choice shape included `finish_reason`, `text`, and a `token_ids`
key. In this response, the `token_ids` value was not a list, so the length
compatibility check was recorded as `not_present` rather than asserting a
count from a null field.

Recorded response shape:

```text
top-level keys: choices, created, id, kv_transfer_params, model, object,
  service_tier, system_fingerprint, usage
first choice keys: finish_reason, index, logprobs, prompt_logprobs,
  prompt_token_ids, routed_experts, stop_reason, text, token_ids
usage keys: completion_tokens, prompt_tokens, prompt_tokens_details,
  total_tokens
```

The probe did not compare or record the generated text as correctness
evidence.

vLLM server log evidence:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 4096
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Loading weights took 32.57 seconds
Model loading took 74.08 GiB memory and 36.290228 seconds
Available KV cache memory: 32.08 GiB
GPU KV cache size: 90,841 tokens
init engine (profile, create kv cache, warmup model) took 8.04 s
Starting vLLM server on http://127.0.0.1:28125
Route: /health, Methods: GET
Route: /v1/models, Methods: GET
Route: /v1/completions, Methods: POST
GET /health HTTP/1.1" 200 OK
GET /v1/models HTTP/1.1" 200 OK
POST /v1/completions HTTP/1.1" 200 OK
```

The server log also recorded first-request Triton JIT warnings during
inference. Those warnings are response-contract observations only, not latency
or throughput evidence.

## Shutdown Behavior

The probe terminated the server process group after contract validation:

```text
terminated: true
killed: false
returncode_after_cleanup: 0
remaining_process_group_pids: []
cleanup.status: passed
```

The server log recorded vLLM shutdown and API server exit. It also emitted a
Python `resource_tracker` shared-memory cleanup warning at shutdown. The probe
still reported no remaining process-group PIDs, and the immediate post-run
selected-GPU memory snapshot returned to the pre-run baseline:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

## Interpretation

The remote H200 vLLM environment can start the local-only server, pass
`/health` and `/v1/models`, return HTTP 200 from one bounded
`/v1/completions` request, satisfy the response-contract checks above, and
shut down with no remaining process-group PIDs under this explicit two-H200
boundary:

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
outer timeout=60m
readiness timeout=2700s
request timeout=180s
```

## Non-Claims

- This is not generated-text correctness evidence.
- This is not tokenizer semantic correctness or prompt correctness evidence.
- This is not 256K context behavior evidence.
- This is not latency, throughput, or production-readiness evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
- This did not commit raw model artifacts, venvs, command dumps, server logs,
  or `tmp/` symlinks.

## Next Gate

The next reviewable gate can make first-request warmup behavior explicit for
the shapes that triggered Triton JIT warnings, or can move to a separate
serving-semantics gate. That later gate needs its own resource plan, timeouts,
expected failure modes, and non-claims.
