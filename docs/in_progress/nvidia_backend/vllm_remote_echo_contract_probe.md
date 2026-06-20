# vLLM Remote H200 Echo-Contract Probe

This note records a bounded remote H200 OpenAI-compatible completion
echo-contract probe for `deepseek-ai/DeepSeek-V4-Flash`. It reuses the
complete repo-relative artifact directory and the local-only vLLM server
boundary from prior gates, then validates response shape for an explicit
`echo=true` completion request. It does not validate generated text.

Raw command output is kept under the gitignored local directory
`tmp/vllm-echo-contract-probe/`.

## Probe Surface

The repo-owned response-contract probe now has an `--echo-contract` mode. It
starts `vllm serve`, binds only to `127.0.0.1`, checks `/health` and
`/v1/models`, sends one bounded non-streaming completion request with explicit
`echo=true`, validates the base structural response contract, validates echo
response shape, terminates the server process group, and reports remaining
process-group PIDs.

Request payload for this gate:

```json
{
  "endpoint": "/v1/completions",
  "payload": {
    "echo": true,
    "max_tokens": 1,
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "n": 1,
    "prompt": "Hello",
    "seed": 0,
    "stream": false,
    "temperature": 0.0,
    "top_p": 1.0
  },
  "limits": {
    "echo": true,
    "max_tokens": 1,
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
explicit echo=true request field
echo response text starts with request prompt
raw generated text is not recorded or checked for correctness
server process group cleanup leaves no remaining PIDs
```

## Inspection

The installed remote vLLM package was inspected before selecting `echo`:

```text
python: 3.12.3
vllm: 0.23.0
torch: 2.11.0+cu130
torch CUDA: 13.0
```

The OpenAI-compatible completion request model is
`vllm.entrypoints.openai.completion.protocol.CompletionRequest`. It exposes
`echo: bool | None = False`, `stop: str | list[str] | None = []`, and
`stop_token_ids: list[int] | None = []`. The non-streaming completion handler
in `vllm.entrypoints.openai.completion.serving` has explicit `request.echo`
handling that returns `prompt_text + output.text` for nonzero `max_tokens`.

The `vllm serve` CLI was also inspected through per-flag help for the local
server boundary flags: `--host`, `--port`, `--max-model-len`,
`--tensor-parallel-size`, `--kv-cache-dtype`,
`--distributed-executor-backend`, and `--enforce-eager`.

## Resource Plan

The remote source tree was refreshed with `--sync` before the dry-run and
before the echo-contract run. The ignored repo-relative artifact directory
and checkout-local `.venv-vllm-probe` were preserved. The synced remote
checkout did not expose usable Git metadata, so the run relies on the `--sync`
command as the source-tree refresh evidence.

Remote tooling and package versions:

```text
GPU: 8 x NVIDIA H200 NVL
driver: 580.126.20
CUDA toolkit: /usr/local/cuda
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

They were selected because this pair passed the prior bounded model-load,
server-health, inference-smoke, response-contract, warmup-shape,
request-shape variation, serving-semantics, and logprobs-contract gates, and
the fresh memory check still showed enough free memory for the 0.78
utilization boundary. This selection is not performance evidence.

Pre-run memory for the selected GPUs:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

The fixed local port was checked before the run:

```text
127.0.0.1:28130 available: yes
```

The planned local-only endpoints were:

```text
GET http://127.0.0.1:28130/health
GET http://127.0.0.1:28130/v1/models
POST http://127.0.0.1:28130/v1/completions
```

Timeouts:

```text
outer timeout: 65m
readiness timeout: 2700s
request timeout: 180s
cleanup timeout: 60s
```

## Bounded Dry-Run

Before loading the model, the same synced remote checkout ran the probe with
`--dry-run`. The dry-run returned `status=planned`,
`generation_attempted=false`, and the explicit `echo=true` payload shown
above.

## Passing Echo-Contract Probe

Core probe command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-vllm-remote-env-artifact \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'set +euo pipefail
RUN_LOG=tmp/vllm-echo-contract-probe/server-28130.log
mkdir -p tmp/vllm-echo-contract-probe
printf "== port availability ==\n"
.venv-vllm-probe/bin/python - <<'"'"'PY'"'"'
import socket
with socket.socket() as s:
    print("127.0.0.1:28130 available:", s.connect_ex(("127.0.0.1", 28130)) != 0)
PY
printf "== versions ==\n"
.venv-vllm-probe/bin/python - <<'"'"'PY'"'"'
import sys
import torch
import vllm
print("python:", sys.version.split()[0])
print("vllm:", vllm.__version__)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
PY
printf "== pre-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
printf "== echo contract json ==\n"
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28130 \
  --server-log "$RUN_LOG" \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 1 --temperature 0.0 --top-p 1.0 \
  --seed 0 --echo-contract
rc=$?
printf "== probe exit code ==\n%s\n" "$rc"
printf "== post-run selected gpu memory ==\n"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
  --format=csv,noheader,nounits -i 1,7
exit "$rc"'
```

Result:

```text
status: passed
exit: 0
elapsed_seconds: 108.998
server_host: 127.0.0.1
server_port: 28130
generation_attempted: true
```

Server command launched by the probe:

```text
.venv-vllm-probe/bin/vllm serve \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 28130 \
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

Base response-contract result:

```text
choice_count: 1
model: deepseek-ai/DeepSeek-V4-Flash
usage.prompt_tokens: 1
usage.completion_tokens: 1
usage.total_tokens: 2
request max_tokens: 1
finish_reason: length
stop_reason: null
token_ids check: not_present
```

Echo-contract result:

```text
echo request field: passed
echo request prompt shape: passed
echo prompt prefix: passed
prompt_chars: 5
text_length_chars: 10
generated_suffix_chars: 5
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

The probe recorded a text digest and length only as opaque response
observations. It did not record the generated text or compare the generated
suffix against an expected answer.

vLLM server log evidence:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 4096
Using DeepSeek's fp8_ds_mla KV cache format.
Loading safetensors checkpoint shards: 100% Completed | 46/46
Loading weights took 31.27 seconds
Model loading took 74.08 GiB memory and 35.612771 seconds
Available KV cache memory: 32.08 GiB
GPU KV cache size: 90,841 tokens
init engine (profile, create kv cache, warmup model) took 9.61 s
Starting vLLM server on http://127.0.0.1:28130
Route: /health, Methods: GET
Route: /v1/models, Methods: GET
Route: /v1/completions, Methods: POST
GET /health HTTP/1.1" 200 OK
GET /v1/models HTTP/1.1" 200 OK
POST /v1/completions HTTP/1.1" 200 OK
```

The server log also recorded first-request Triton JIT warnings during
inference. Those warnings are echo-contract observations only, not latency or
throughput evidence.

## Shutdown Behavior

The probe terminated the server process group after contract validation:

```text
cleanup status: passed
terminated: true
killed: false
remaining_process_group_pids: []
returncode_after_cleanup: 0
```

The server log recorded vLLM shutdown and API server exit. It also emitted a
process-manager force-kill warning plus a PyTorch distributed store warning
from one worker during shutdown. The probe still reported no remaining
process-group PIDs, and the immediate post-run selected-GPU memory snapshot
returned to the pre-run baseline:

```text
GPU 1: 141773 MiB free / 143771 MiB total
GPU 7: 116662 MiB free / 143771 MiB total
```

## Non-Claims

- This is not generated-text correctness evidence.
- This is not tokenizer semantic correctness evidence.
- This is not prompt correctness evidence.
- This is not token identity or logprob value correctness evidence.
- This is not 256K context evidence.
- This is not latency or throughput evidence.
- This is not production-readiness evidence.
- This is not simpler-nv or vLLM kernel integration evidence.
