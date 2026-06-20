# vLLM Remote H200 128K Context Health Probe

This note records a bounded remote H200 vLLM server-health/model-list probe
for `deepseek-ai/DeepSeek-V4-Flash` with `--max-model-len 131072`. It reuses
the complete repo-relative artifact directory and the local-only server
boundary from prior gates. It does not send a long prompt, run generation, or
validate generated text.

Raw command output is kept under the gitignored local directory
`tmp/vllm-128k-context-health-probe/`.

## Probe Surface

The repo-owned server-health probe starts `vllm serve`, binds only to
`127.0.0.1`, checks `/health` and `/v1/models`, emits structured JSON,
terminates the server process group, and reports remaining process-group PIDs.

Contract checks:

```text
HTTP 200 from /health
HTTP 200 from /v1/models
served model id is deepseek-ai/DeepSeek-V4-Flash
model-list max_model_len is 131072
generation_attempted is false
server process group cleanup leaves no remaining PIDs
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the run. The ignored
repo-relative artifact directory and checkout-local `.venv-vllm-probe` were
preserved. The synced remote checkout did not expose usable Git metadata, so
the run relies on the `--sync` command as the source-tree refresh evidence.

Remote tooling and package versions:

```text
GPU: 2 selected NVIDIA H200 NVL devices from an 8-GPU host
driver: 580.126.20
compute capability: 9.0
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

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28133 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28133/health
GET http://127.0.0.1:28133/v1/models
```

Timeouts:

```text
outer timeout: 50m
readiness timeout: 2700s
poll interval: 10s
cleanup timeout: 60s
```

## Passing 128K Health Probe

Core probe command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'RUN_DIR=tmp/vllm-128k-context-health-probe
RUN_LOG="$RUN_DIR/server-28133.log"
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28133 \
  --server-log "$RUN_LOG" \
  --max-model-len 131072 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 101.546
server_host: 127.0.0.1
server_port: 28133
generation_attempted: false
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28133 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --tokenizer-mode deepseek_v4 \
  --max-model-len 131072 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

Endpoint results:

```text
/health: HTTP 200 after 11 polling attempts
/v1/models: HTTP 200
model id: deepseek-ai/DeepSeek-V4-Flash
model max_model_len: 131072
```

vLLM server log evidence:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 131072
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Loading weights took 32.78 seconds
Model loading took 74.08 GiB memory and 37.450321 seconds
Available KV cache memory: 32.08 GiB
GPU KV cache size: 1,247,687 tokens
Maximum concurrency for 131,072 tokens per request: 9.52x
init engine (profile, create kv cache, warmup model) took 9.90 s
Starting vLLM server on http://127.0.0.1:28133
Route: /health, Methods: GET
Route: /v1/models, Methods: GET
GET /health HTTP/1.1" 200 OK
GET /v1/models HTTP/1.1" 200 OK
```

## Shutdown Behavior

The probe terminated the server process group after the endpoint checks:

```text
terminated: true
killed: false
returncode_after_cleanup: 0
remaining_process_group_pids: []
cleanup.status: passed
```

The immediate post-run `nvidia-smi` snapshot showed the selected GPUs back at
or near their pre-run memory baseline:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116670 MiB free / 143771 MiB total
```

The final server log emitted Python `resource_tracker` cleanup warnings for
semaphore and shared-memory objects at shutdown. The probe still reported no
remaining process-group PIDs, and selected GPU memory returned to the baseline
above.

## Interpretation

The remote H200 vLLM environment can start a local-only OpenAI-compatible
vLLM server for `deepseek-ai/DeepSeek-V4-Flash` from the complete
repo-relative artifacts under this explicit two-H200 boundary:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
max_model_len=131072
dtype=bfloat16
quantization=deepseek_v4_fp8
kv_cache_dtype=fp8
gpu_memory_utilization=0.78
enforce_eager=true
distributed_executor_backend=mp
outer timeout=50m
readiness timeout=2700s
```

The server bound to `127.0.0.1`, returned HTTP 200 from `/health`, returned
the served model with `max_model_len=131072` from `/v1/models`, and shut down
with no remaining process-group PIDs reported by the probe. This is a 128K
server-health/model-list capacity gate only.

## Non-Claims

- This did not send a long prompt.
- This is not generated-text correctness evidence.
- This is not tokenizer semantic correctness or prompt correctness evidence.
- This is not stop, logprob, token identity, or output-text correctness
  evidence.
- This is not 256K context behavior evidence.
- This is not latency, throughput, or production-readiness evidence.
- This is not broad determinism evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
- This did not commit raw model artifacts, venvs, command dumps, server logs,
  or `tmp/` symlinks.

## Next Gate

The next reviewable context-capacity gate can attempt 256K
server-health/model-list readiness under the same local-only contract without
sending a long prompt. That later gate should stay separate from generated
text correctness, throughput, latency, and production-readiness claims.
