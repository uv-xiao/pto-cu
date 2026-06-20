# vLLM Remote H200 Server Health Probe

This note records a bounded remote H200 vLLM server startup and health probe
for `deepseek-ai/DeepSeek-V4-Flash`. It uses the complete repo-relative
artifact directory from prior gates. It starts a local-only OpenAI-compatible
server, checks readiness endpoints, and shuts it down. It does not run
generation or validate generated text.

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-server-health-probe/`.

## CLI Inspection

The installed vLLM 0.23.0 server surface was inspected before choosing the
command. The remote vLLM probe environment exposes the console entry point:

```text
.venv-vllm-probe/bin/vllm serve [model_tag] [options]
```

The inspected `vllm serve --help` and OpenAI API server parser expose the
required flags used by the probe:

```text
--host
--port
--served-model-name
--tokenizer
--tokenizer-mode
--max-model-len
--tensor-parallel-size
--dtype
--quantization
--kv-cache-dtype
--gpu-memory-utilization
--distributed-executor-backend
--enforce-eager
```

## Resource Plan

The remote source tree was refreshed with `--sync` before the resource-plan
capture and before the server run. The ignored repo-relative artifact
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
model-load gate and still had enough free memory for the 0.78 utilization
boundary. This selection is not performance evidence.

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28123 available: yes
```

The planned health endpoints were:

```text
GET http://127.0.0.1:28123/health
GET http://127.0.0.1:28123/v1/models
```

## Passing Server Probe

Exact command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set +euo pipefail
RUN_LOG=tmp/vllm-server-health-probe/server-28123-corrected.log
mkdir -p tmp/vllm-server-health-probe
printf "== pre-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== server probe json ==\n"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28123 \
  --server-log "$RUN_LOG" \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
rc=$?
printf "== probe exit code ==\n%s\n" "$rc"
printf "== post-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== server log tail ==\n"
tail -n 240 "$RUN_LOG" 2>/dev/null
exit "$rc"'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 114.348
generation_attempted: false
server_host: 127.0.0.1
server_port: 28123
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28123 \
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
```

vLLM server log evidence:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 4096
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Loading weights took 32.74 seconds
Model loading took 74.08 GiB memory and 38.516295 seconds
Available KV cache memory: 32.08 GiB
GPU KV cache size: 90,841 tokens
init engine (profile, create kv cache, warmup model) took 11.45 s
Starting vLLM server on http://127.0.0.1:28123
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

The server log recorded vLLM shutdown and API server exit:

```text
[shutdown] EngineCore: trigger received signal=SIGTERM
[shutdown] API server: shutdown triggered
Application shutdown complete.
Finished server process
```

The immediate post-run `nvidia-smi` snapshot showed the selected GPUs back at
their pre-run baseline:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The final server log also emitted a Python `resource_tracker` shared-memory
cleanup warning at shutdown. The probe still reported no remaining process
group PIDs, and selected GPU memory returned to the baseline above.

## Interpretation

The remote H200 vLLM environment can start a local-only OpenAI-compatible
vLLM server for `deepseek-ai/DeepSeek-V4-Flash` from the complete
repo-relative artifacts under this explicit two-H200 boundary:

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
outer timeout=50m
readiness timeout=2700s
```

The server bound to `127.0.0.1`, returned HTTP 200 from `/health`, returned
the served model from `/v1/models`, and shut down with no remaining process
group PIDs reported by the probe. GPU memory returned to the pre-run baseline
in a follow-up check after the command completed.

## Non-Claims

- This is not generated-text correctness evidence.
- This is not tokenizer semantic correctness or prompt correctness evidence.
- This is not 256K context behavior evidence.
- This is not latency, throughput, or production-readiness evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
- This did not commit raw model artifacts, venvs, command dumps, server logs,
  or `tmp/` symlinks.

## Next Gate

The next reviewable gate can be an explicitly bounded one-token inference
smoke against this local-only server if serving semantics are needed. That gate
should be documented as inference-smoke evidence only, not correctness,
throughput, latency, long-context, or production-readiness evidence.
