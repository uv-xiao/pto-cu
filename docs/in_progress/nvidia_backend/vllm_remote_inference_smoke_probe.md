# vLLM Remote H200 Inference Smoke Probe

This note records a bounded remote H200 one-token inference smoke for
`deepseek-ai/DeepSeek-V4-Flash`. It reuses the complete repo-relative
artifact directory and the local-only server boundary from the prior server
health gate, then sends one OpenAI-compatible completion request. It does not
validate generated text.

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-inference-smoke-probe/`.

## Probe Surface

A repo-owned smoke probe script starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, sends one bounded inference
request, records response shape rather than expected text, terminates the
server process group, and reports remaining process-group PIDs.

Default inference request:

```json
{
  "endpoint": "/v1/completions",
  "payload": {
    "max_tokens": 1,
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "prompt": "Hello",
    "stream": false,
    "temperature": 0.0
  },
  "limits": {
    "max_tokens": 1,
    "prompt_chars": 5,
    "stream": false
  }
}
```

The script rejects `--max-tokens` values above 1. It also supports
`/v1/chat/completions`, but this gate used `/v1/completions`.

## Resource Plan

The remote source tree was refreshed with `--sync` before the resource-plan
capture and before the inference run. The ignored repo-relative artifact
directory and checkout-local `.venv-vllm-probe` were preserved.

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
model-load and server-health gates, and the fresh memory check showed enough
free memory for the 0.78 utilization boundary. This selection is not
performance evidence.

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28124 available: yes
```

The planned endpoints were:

```text
GET http://127.0.0.1:28124/health
GET http://127.0.0.1:28124/v1/models
POST http://127.0.0.1:28124/v1/completions
```

Request timeout:

```text
180 seconds
```

## Passing Inference Smoke

Exact command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set +euo pipefail
RUN_LOG=tmp/vllm-inference-smoke-probe/server-28124.log
mkdir -p tmp/vllm-inference-smoke-probe
printf "== pre-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== inference smoke json ==\n"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 55m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_inference_smoke_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28124 \
  --server-log "$RUN_LOG" \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --endpoint /v1/completions --prompt Hello --max-tokens 1 \
  --temperature 0.0
rc=$?
printf "== probe exit code ==\n%s\n" "$rc"
printf "== post-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== server log tail ==\n"
tail -n 260 "$RUN_LOG" 2>/dev/null
exit "$rc"'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 213.558
server_host: 127.0.0.1
server_port: 28124
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28124 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer-mode deepseek_v4 \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

Readiness results:

```text
/health: HTTP 200 after 12 polling attempts
/v1/models: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 4096
```

Inference request result:

```text
endpoint: /v1/completions
HTTP status: 200
request max_tokens: 1
prompt chars: 5
stream: false
response choice_count: 1
```

Recorded response shape:

```text
top-level keys: choices, created, id, kv_transfer_params, model, object,
  service_tier, system_fingerprint, usage
first choice keys: finish_reason, index, logprobs, prompt_logprobs,
  prompt_token_ids, routed_experts, stop_reason, text, token_ids
usage keys: completion_tokens, prompt_tokens, prompt_tokens_details,
  total_tokens
```

The probe did not compare or record the generated token text as correctness
evidence.

vLLM server log evidence:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 4096
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Loading weights took 36.73 seconds
Model loading took 74.08 GiB memory and 40.758206 seconds
Available KV cache memory: 32.08 GiB
GPU KV cache size: 90,841 tokens
init engine (profile, create kv cache, warmup model) took 8.12 s
Starting vLLM server on http://127.0.0.1:28124
Route: /health, Methods: GET
Route: /v1/models, Methods: GET
Route: /v1/completions, Methods: POST
GET /health HTTP/1.1" 200 OK
GET /v1/models HTTP/1.1" 200 OK
POST /v1/completions HTTP/1.1" 200 OK
```

The server log also recorded first-inference JIT/TileLang compilation warnings
and an engine shared-memory wait while inference was in progress. Those are
inference-smoke observations only and are not latency or throughput evidence.

## Shutdown Behavior

The probe terminated the server process group after the inference request:

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

The remote H200 vLLM environment can start the already proven local-only
server, pass `/health` and `/v1/models`, send one bounded `/v1/completions`
request with `max_tokens=1`, receive HTTP 200 with one response choice, and
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
outer timeout=55m
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

The next reviewable gate can decide whether to reduce first-request
JIT/TileLang compilation surprises with an explicit warmup shape, or move to a
separate serving-semantics check. That later gate needs its own resource plan,
timeouts, expected failure modes, and non-claims.
