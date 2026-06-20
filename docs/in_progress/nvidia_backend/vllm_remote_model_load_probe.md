# vLLM Remote H200 Model Load Probe

This note records a bounded remote H200 vLLM model-load and engine
initialization probe for `deepseek-ai/DeepSeek-V4-Flash`. It uses the complete
repo-relative artifact directory from the prior artifact gate. It does not
start a vLLM server, run inference, validate generated text, measure latency or
throughput, or exercise 256K context behavior.

Raw command output is kept under the gitignored local directory
`tmp/vllm-remote-model-load-probe/`.

## Resource Plan

The remote source tree was refreshed with `--sync` before each command. The
ignored repo-relative artifact directory and checkout-local `.venv-vllm-probe`
were preserved.

The installed vLLM API was inspected from the vLLM probe environment before
loading. The relevant `vllm.LLM` and `EngineArgs` surface supports:

```text
tensor_parallel_size
dtype
quantization
kv_cache_dtype
max_model_len
gpu_memory_utilization
enforce_eager
distributed_executor_backend
trust_remote_code
```

Remote tooling and package versions:

```text
GPU: 8 x NVIDIA H200 NVL
compute capability: 9.0
driver: 580.126.20
CUDA toolkit: /usr/local/cuda, nvcc 12.8 V12.8.61
python: 3.12.3
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
```

The selected physical GPUs were 1 and 7, exposed to vLLM as exactly two visible
devices:

```text
CUDA_VISIBLE_DEVICES=1,7
tensor_parallel_size=2
```

The topology report showed GPU 1 and GPU 7 connected through `SYS`, not NVLink.
They were selected because they were the only observed pair with enough free
memory for a two-H200 load attempt. This selection is not performance evidence.

Pre-attempt memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116562 MiB free / 143771 MiB total
```

## First Attempt

The first bounded attempt used:

```text
max_model_len=4096
tensor_parallel_size=2
dtype=bfloat16
quantization=deepseek_v4_fp8
kv_cache_dtype=auto
gpu_memory_utilization=0.82
enforce_eager=true
distributed_executor_backend=mp
timeout=45m
```

It failed during worker initialization. The raw log recorded two actionable
blockers:

```text
Free memory on device cuda:1 (112.72/139.8 GiB) on startup is less than
desired GPU memory utilization (0.82, 114.64 GiB).

DeepseekV4 FlashMLA fp8 layout only supports fp8 kv-cache, got auto
```

This attempt did not establish engine initialization.

## Passing Attempt

The retry changed only the two settings directly indicated by the first
attempt: `kv_cache_dtype=fp8` and `gpu_memory_utilization=0.78`.

Exact command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set -euo pipefail
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 45m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_model_load_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager'
```

Result:

```text
status: passed
load_attempted: true
loaded_engine_class: LLM
llm_engine_class: LLMEngine
elapsed_seconds: 605.411
exit: 0
```

Artifact state in the passing run:

```text
indexed_tensors: 69187
indexed_shards: 46
present_shards: 46
missing_shards: 0
present_bytes: 159617149040
index_total_size: 159609485896
```

vLLM load/init evidence from the log:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 4096
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Loading weights took 36.95 seconds
Model loading took 74.35 GiB memory and 42.924824 seconds
Available KV cache memory: 29.07 GiB
GPU KV cache size: 82,301 tokens
Maximum concurrency for 4,096 tokens per request: 20.09x
init engine (profile, create kv cache, warmup model) took 516.40 s
```

The run also recorded that eager mode disabled torch compile and CUDAGraphs,
while vLLM still performed DeepGEMM/TileLang initialization and warmup inside
the engine-initialization command.

In-process GPU memory snapshots for the selected physical GPUs:

```text
before: GPU 1 used 1383 MiB, free 141773 MiB
before: GPU 7 used 26595 MiB, free 116562 MiB
after:  GPU 1 used 116087 MiB, free 27069 MiB
after:  GPU 7 used 141199 MiB, free 1958 MiB
```

A post-exit snapshot showed the selected GPUs returned to their pre-run
baseline memory:

```text
GPU 1 used 1383 MiB, free 141773 MiB
GPU 7 used 26495 MiB, free 116662 MiB
```

## Interpretation

The remote H200 vLLM environment can initialize a `vllm.LLM` /
`LLMEngine` for `deepseek-ai/DeepSeek-V4-Flash` from the complete
repo-relative artifact directory on exactly two H200 GPUs under the recorded
resource boundary:

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
timeout=45m
```

The successful command loaded all 46 safetensors shards and completed vLLM
engine initialization. The command did not start a server or run generation.

## Non-Claims

- This is not vLLM server startup or server health evidence.
- This is not inference, generated-text correctness, prompt correctness, or
  tokenizer semantic correctness evidence.
- This is not 256K context behavior evidence.
- This is not latency, throughput, or production-readiness evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
- This did not commit raw model artifacts, venvs, command dumps, or `tmp/`
  symlinks.

## Next Gate

The next reviewable gate should be a separately bounded vLLM server startup
and health probe, or an explicitly documented one-token smoke if server startup
requires generation to prove readiness. That gate should reuse the explicit
resource boundary above or record a new one before running.
