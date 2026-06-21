# pypto-serving Source Contract H200 Evidence

This note records the first H200 smoke that exercises the actual
`pypto-serving` server source rather than only the local lookalike HTTP
fixture.

## Contract

The source file under test is:

```text
tmp/sources/repos/hw-native-sys/pypto-serving/python/core/server.py
```

The smoke imports `python.core.server.create_serving_app` from the cloned
source tree and uses the real `ServingServer` routes:

- `/health`
- `/v1/models`
- `/v1/completions`
- `/v1/chat/completions`

`PyptoServingSourceAsyncEngineAdapter` adapts the synthetic simpler-nv engine
to the server's async `add_request(...)` contract. The adapter returns the
token-output shape expected by the actual server while preserving PTO debug
evidence outside the server response.

The tracked entry points are:

- `create_pypto_serving_source_app(...)`
- `run_pypto_serving_source_completion_fixture(...)`
- `run_pypto_serving_source_chat_completion_fixture(...)`
- `run_pypto_serving_source_stream_completion_fixture(...)`
- `run_pypto_serving_source_stream_chat_completion_fixture(...)`
- CLI flag: `--pypto-serving-source`
- CLI flag: `--pypto-serving-source-chat`
- CLI flag: `--pypto-serving-source-stream`
- CLI flag: `--pypto-serving-source-chat-stream`
- CLI flag: `--pypto-serving-vllm-compat`
- CLI flag:
  `--kernel-launcher {cuda-seed,gluon-moe-expert,gluon-topk-sampling,persistent-moe-dispatch-combine}`

## Source Chat Contract

`run_pypto_serving_source_chat_completion_fixture(...)` imports the same
cloned source `create_serving_app(...)` and posts a bounded non-streaming
OpenAI-style `messages` list to `/v1/chat/completions`. The request includes
one small user message and `max_tokens=2`; no raw private paths or model
artifacts are recorded.

The fixture records review-safe response shape rather than raw serving logs:
route, HTTP status code, top-level `object`, assistant message role/content
shape, mapped finish reason, `pto_status`, generated PTO token IDs, and
`pto_launch_count`.

## Source Streaming Contract

`run_pypto_serving_source_stream_completion_fixture(...)` and
`run_pypto_serving_source_stream_chat_completion_fixture(...)` post the same
bounded source-route requests with `stream: true`. The synthetic adapter
yields cumulative text per generated PTO token, so the real source server's
SSE delta slicing is exercised:

- completion route `/v1/completions`: cumulative adapter outputs `N`, `NV`
  produce streamed text chunks `N`, `V`;
- chat route `/v1/chat/completions`: cumulative adapter outputs `N`, `NV`
  produce assistant delta chunks `N`, `V`;
- both routes emit the terminal `[DONE]` event after the final chunk.

The streaming fixtures record review-safe summaries: route, HTTP status code,
`stream: true`, `event_count`, `chunk_count`, terminal `[DONE]` presence,
assembled completion text or assistant deltas, mapped finish reason,
`pto_status`, generated PTO token IDs, and `pto_launch_count`.

Local source streaming result shape:

```text
server: pypto-serving-source
route: /v1/completions
stream: true
status_code: 200
event_count: 3
chunk_count: 2
done_seen: true
assembled_text: NV
finish_reason: length
pto_status: passed
pto_token_ids: [1, 2]
pto_launch_count: 2

server: pypto-serving-source
route: /v1/chat/completions
stream: true
status_code: 200
event_count: 3
chunk_count: 2
done_seen: true
assistant_deltas: [N, V]
assembled_assistant_text: NV
finish_reason: length
pto_status: passed
pto_token_ids: [1, 2]
pto_launch_count: 2
```

## Local Verification

Focused source-contract tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
    -q -k 'pypto_serving_source'
```

Result:

```text
9 passed, 18 deselected
```

Full shim tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
27 passed
```

## H200 Source Sync

The generic remote runner excludes `tmp/`, so the `pypto-serving` source clone
was synced explicitly before the H200 run:

```bash
<remote-shell> <h200-host> 'mkdir -p <remote-pto-cu>/tmp/sources/repos/hw-native-sys'
rsync -a --delete --exclude=.git \
  tmp/sources/repos/hw-native-sys/pypto-serving/ \
  <h200-host>:<remote-pto-cu>/tmp/sources/repos/hw-native-sys/pypto-serving/
```

## H200 Completion Evidence

Environment:

```text
machine: <h200-host>
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
nvcc: Build cuda_12.8.r12.8/compiler.35404655_0
source sync: local working tree synced with --sync; pypto-serving source
  checkout synced explicitly because tmp/ is excluded by the generic runner
```

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source --require-cuda \
      --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90'
```

Result:

```text
server: pypto-serving-source
route: /v1/completions
status: passed
status_code: 200
object: text_completion
text: NV
finish_reason: length
pto_status: passed
pto_token_ids: [1, 2]
pto_launch_count: 2
```

## H200 Chat Evidence

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source-chat --require-cuda \
      --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90'
```

Result:

```text
server: pypto-serving-source
route: /v1/chat/completions
status: passed
status_code: 200
object: chat.completion
assistant_message: {role: assistant, content: NV}
finish_reason: length
pto_status: passed
pto_token_ids: [1, 2]
pto_launch_count: 2
```

## vLLM Compatibility Contract

`--pypto-serving-vllm-compat` emits a local JSON summary that compares the
four source-route fixtures against the structural OpenAI-compatible fields
already exercised by the vLLM DeepSeek serving probes. It covers:

- route, HTTP 200, object/model shape, text or assistant delta;
- finish reason availability;
- usage presence for non-streaming responses;
- terminal `[DONE]` presence for streaming responses.

The summary is intentionally a source-route compatibility gate for the
synthetic adapter. It records non-claims explicitly: not tokenizer semantics,
not logprob values, not stop-token semantics, not production readiness,
not throughput, not latency, not real DeepSeek weights, and
not simpler-nv/vLLM kernel integration evidence.
The current cloned source routes do not return non-streaming `usage`, so the
summary records usage presence as a comparison gap rather than synthesizing it.

## Generated Gluon MoE Expert Launch Contract

The source-route fixtures now accept an explicit launcher selection. The
default remains the CUDA seed launch path, which emits `op: add` and preserves
the previous skip-safe behavior. Passing
`--kernel-launcher gluon-moe-expert` routes the same synthetic
`pypto-serving` request through the existing generated Gluon MoE expert
correctness harness:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
    --pypto-serving-source --kernel-launcher gluon-moe-expert \
    --prompt hello --max-new-tokens 1 --device 0 --arch compute_90
```

The launch result records review-safe generated-kernel metadata under
`pto_launch_results`, including:

```text
launch_kind: gluon-moe-expert
kernel_name: moe_expert_affine_f32
phase: prefill
status: passed|skipped|failed
shape.n: 16
source_sha256: <generated-source-digest-when-available>
artifact.source_path: tmp/...
artifact.manifest_path: tmp/...
```

The generated launch mode calls
`examples/cuda/gluon_moe_expert_affine.py::run_moe_expert_correctness(...)`.
Without local CUDA, torch CUDA, or Triton Gluon availability, that harness
returns a structured skip; with `--require-cuda`, the CLI returns non-zero for
that skip. The generated artifacts stay under ignored repo-relative `tmp/`
directories.

Local generated source-route verification:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
    --pypto-serving-source --kernel-launcher gluon-moe-expert \
    --require-cuda --prompt hello --max-new-tokens 1 \
    --device 0 --arch compute_90
```

Result:

```text
server: pypto-serving-source
route: /v1/completions
status: passed
status_code: 200
text: N
pto_status: passed
pto_launch_count: 1
launch_kind: gluon-moe-expert
kernel_name: moe_expert_affine_f32
phase: prefill
shape.n: 16
source_sha256: 38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842
```

Local unit coverage proves the completion, chat, streaming completion, and
streaming chat source fixtures can all receive this generated launcher hook.

## Generated Gluon Top-K Sampling Launch Contract

Passing `--kernel-launcher gluon-topk-sampling` routes the same synthetic
`pypto-serving` request through
`examples/cuda/gluon_topk_sampling.py::run_topk_sampling_correctness(...)`.
The source-route launcher records `launch_kind: gluon-topk-sampling`,
`kernel_name: topk_sampling_f32`, phase, status, shape, generated
artifact/source digest metadata, validation metadata when present, and the
same non-claim boundary as the standalone Top-K harness.

H200 evidence for this selected launcher is recorded in
`pypto_serving_topk_sampling_launcher_h200.md`. It is a serving-route
launcher/probe for a generated Top-K sampling correctness gate, not
FlashInfer integration, tokenizer semantics, generated text correctness,
DeepSeek serving readiness, or production serving evidence.

## H200 Generated Gluon MoE Source-Route Matrix

The H200 pass used tree sync rather than a remote Git refresh. The tracked
working tree was synced with `--sync`; the ignored `pypto-serving` source
clone was synced explicitly because the generic runner excludes `tmp/`. The
route commands then reused the synced checkout and the same source clone.

Commit tested:

```text
pto-cu: b66e1ece
pypto-serving source clone: 0b0d8a0
remote Git refresh: not used
```

Environment:

```text
machine: <h200-host>
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
nvcc: Build cuda_12.8.r12.8/compiler.35404655_0
python environment: <remote-gluon-venv>
torch: 2.8.0+cu128
torch CUDA: 12.8
triton: 3.7.1
stderr caveat: no Torch/NumPy compatibility warning was printed
```

Compact route-matrix command shape:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-vllm-compat --kernel-launcher gluon-moe-expert \
      --require-cuda --prompt hello --max-new-tokens 1 \
      --device 0 --arch compute_90'
```

Compatibility summary result:

```text
server: pypto-serving-source
status: passed
comparison_baseline: vllm-openai-compatible-deepseek
checked_fields:
  route
  http_status_200
  model_or_stream_object_shape
  choice_text_or_message_delta_presence
  finish_reason
  usage_presence_when_non_streaming
  sse_done_presence_when_streaming

fixture: completions
route: /v1/completions
stream: false
status: passed
status_code: 200
object: text_completion
text_present: true
finish_reason: length
usage_keys: []
pto_status: passed
pto_launch_count: 1

fixture: chat_completions
route: /v1/chat/completions
stream: false
status: passed
status_code: 200
object: chat.completion
message_role: assistant
message_content: true
finish_reason: length
usage_keys: []
pto_status: passed
pto_launch_count: 1

fixture: stream_completions
route: /v1/completions
stream: true
status: passed
status_code: 200
event_count: 2
chunk_count: 1
done_seen: true
assembled_text: N
finish_reason: length
pto_status: passed
pto_launch_count: 1

fixture: stream_chat_completions
route: /v1/chat/completions
stream: true
status: passed
status_code: 200
event_count: 2
chunk_count: 1
done_seen: true
assembled_assistant_text: N
finish_reason: length
pto_status: passed
pto_launch_count: 1
```

The structural summary intentionally records the non-streaming source-route
`usage` gap instead of synthesizing usage fields.

The per-route commands below recorded generated-kernel launch metadata. Each
used the same command shape with one route flag substituted:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    <remote-gluon-venv>/bin/python \
      examples/cuda/pypto_serving_nv_shim.py \
      <source-route-flag> --kernel-launcher gluon-moe-expert \
      --require-cuda --prompt hello --max-new-tokens 1 \
      --device 0 --arch compute_90'
```

The route flags were:

- `--pypto-serving-source`
- `--pypto-serving-source-chat`
- `--pypto-serving-source-stream`
- `--pypto-serving-source-chat-stream`

### `/v1/completions`

```text
server: pypto-serving-source
route: /v1/completions
status: passed
status_code: 200
object: text_completion
text: N
finish_reason: length
pto_status: passed
pto_token_ids: [1]
pto_launch_count: 1
launch_kind: gluon-moe-expert
kernel_name: moe_expert_affine_f32
phase: prefill
shape.n: 16
source_sha256: 38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842
max_abs_error: 1.1920928955078125e-07
```

### `/v1/chat/completions`

```text
server: pypto-serving-source
route: /v1/chat/completions
status: passed
status_code: 200
object: chat.completion
assistant_message: {role: assistant, content: N}
finish_reason: length
pto_status: passed
pto_token_ids: [1]
pto_launch_count: 1
launch_kind: gluon-moe-expert
kernel_name: moe_expert_affine_f32
phase: prefill
shape.n: 16
source_sha256: 38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842
max_abs_error: 1.1920928955078125e-07
```

### Streaming `/v1/completions`

```text
server: pypto-serving-source
route: /v1/completions
stream: true
status: passed
status_code: 200
event_count: 2
chunk_count: 1
done_seen: true
assembled_text: N
finish_reason: length
pto_status: passed
pto_token_ids: [1]
pto_launch_count: 1
launch_kind: gluon-moe-expert
kernel_name: moe_expert_affine_f32
phase: prefill
shape.n: 16
source_sha256: 38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842
max_abs_error: 1.1920928955078125e-07
```

### Streaming `/v1/chat/completions`

```text
server: pypto-serving-source
route: /v1/chat/completions
stream: true
status: passed
status_code: 200
event_count: 2
chunk_count: 1
done_seen: true
assistant_deltas: [N]
assembled_assistant_text: N
finish_reason: length
pto_status: passed
pto_token_ids: [1]
pto_launch_count: 1
launch_kind: gluon-moe-expert
kernel_name: moe_expert_affine_f32
phase: prefill
shape.n: 16
source_sha256: 38bb58f3f019a6eefb4016ff180b988f0b1532e5eee4bade5e49d7f57038b842
max_abs_error: 1.1920928955078125e-07
```

This is generated-kernel source-route contract evidence for the synthetic
adapter. It is not DeepSeek-V4-Flash serving, not vLLM plugin integration, not
FlashInfer integration, not production readiness, not throughput or latency,
not distributed serving, and not fused MoE dispatch/combine serving readiness.

## Persistent MoE Dispatch/Combine Launch Contract

The source-route launcher selection also accepts
`--kernel-launcher persistent-moe-dispatch-combine`. That mode routes the same
synthetic `pypto-serving` request through
`examples/cuda/persistent_moe_dispatch_combine.py::run_moe_dispatch_combine(...)`.
The shim does not duplicate the persistent-device graph logic; it only adapts
the existing example's structured result into `pto_launch_results`.

The launch result records bounded review-safe metadata:

```text
launch_kind: persistent-moe-dispatch-combine
phase: prefill
status: passed|skipped|failed
dag_shape: graph_descriptor_moe_dispatch_combine
shape.n: 16
completed_count: <completed-count-when-present>
max_abs_error: <max-absolute-error-when-present>
scheduler_error_summary: <device-scheduler-errors-when-present>
gluon_expert_bridge.source_sha256: <bridge-digest-when-present>
task_body_digest.func_id: 12
task_body_digest.source_sha256: <task-body-digest-when-present>
```

Local unit coverage proves the completion, chat, streaming completion, and
streaming chat source fixtures can all receive this persistent launcher hook.
The launcher remains skip-safe when CUDA, runtime build, or device
prerequisites are missing; with `--require-cuda`, a selected-launcher skip
returns non-zero through the existing CLI status handling.

Local focused verification:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
    -q -k 'persistent_moe_dispatch_combine_launcher or \
persistent_launcher_can_run_through_source_route_fixtures or \
persistent_moe_cli_mode_outputs_launch_metadata'
```

Result:

```text
3 passed, 34 deselected
```

This is source-route launch contract evidence for the synthetic adapter. It is
not DeepSeek-V4-Flash serving, not real model loading, not tokenizer
semantics, not vLLM plugin integration, not throughput or latency evidence,
not distributed expert parallelism, and not production serving readiness.

## H200 Persistent MoE Source-Route Matrix

The H200 pass used tree sync rather than a remote Git refresh. The tracked
working tree was synced with `--sync`; the ignored `pypto-serving` source
clone was synced explicitly because the generic runner excludes `tmp/`. A
checkout-local remote virtual environment was created with
`--system-site-packages` for source-route dependencies.

Environment:

```text
machine: <h200-host>
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
python: 3.12.3
torch: 2.12.1
triton: 3.7.1
remote Git refresh: not used
source sync: --sync plus explicit pypto-serving source clone sync
stderr caveat: FastAPI TestClient printed a StarletteDeprecationWarning
```

The per-route commands used this shape with one route flag substituted:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      <source-route-flag> \
      --kernel-launcher persistent-moe-dispatch-combine \
      --require-cuda --prompt hello --max-new-tokens 1 \
      --device 0 --arch compute_90'
```

The route flags were:

- `--pypto-serving-source`
- `--pypto-serving-source-chat`
- `--pypto-serving-source-stream`
- `--pypto-serving-source-chat-stream`

Each route produced:

```text
status: passed
status_code: 200
pto_status: passed
pto_token_ids: [1]
pto_launch_count: 1
launch_kind: persistent-moe-dispatch-combine
phase: prefill
dag_shape: graph_descriptor_moe_dispatch_combine
shape.n: 16
completed_count: 5
max_abs_error: 0.0
scheduler_error_summary: {count: 0, code: 0, task_id: 0}
gluon_expert_bridge.kernel_name: moe_expert_affine_f32
gluon_expert_bridge.source_sha256:
  7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f
task_body_digest.func_id: 12
task_body_digest.source_sha256:
  7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f
device_scheduler_errors: {count: 0, code: 0, task_id: 0}
fanin_remaining: [0, 0, 0, 0, 0]
```

Route-specific response summaries:

```text
route: /v1/completions
object: text_completion
text: N
finish_reason: length

route: /v1/chat/completions
object: chat.completion
assistant_message: {role: assistant, content: N}
finish_reason: length

route: /v1/completions
stream: true
event_count: 2
chunk_count: 1
done_seen: true
assembled_text: N
finish_reason: length

route: /v1/chat/completions
stream: true
event_count: 2
chunk_count: 1
done_seen: true
assistant_deltas: [N]
assembled_assistant_text: N
finish_reason: length
```

## H200 Persistent MoE Aggregate Compatibility Evidence

The aggregate `--pypto-serving-vllm-compat` command now runs the four source
routes as isolated child route processes when
`--kernel-launcher persistent-moe-dispatch-combine` is selected. This avoids a
same-process repeated `ctypes.CDLL(...)` runtime binding crash that previously
ended the SSH session before JSON. The source server fixture also loads the
cloned `python/core/server.py` directly with minimal server-route stubs, so
the synthetic adapter contract no longer requires importing the source
package `python.core.__init__` or its Torch-backed real model engine.

The H200 pass used tree sync rather than a remote Git refresh. The tracked
working tree was synced with `--sync`; the ignored `pypto-serving` source
clone was synced explicitly because the generic runner excludes `tmp/`. A
checkout-local remote virtual environment was created with
`--system-site-packages`, and `fastapi`/`httpx` were installed into that
remote venv for the source-route TestClient dependency.

Environment:

```text
machine: <h200-host>
gpu: NVIDIA H200 NVL, compute capability 9.0, 143771 MiB
driver: 580.126.20
CUDA_HOME: /usr/local/cuda
nvcc: Build cuda_12.8.r12.8/compiler.35404655_0
python: 3.12.3
source sync: --sync plus explicit pypto-serving source clone sync
stderr caveat: FastAPI TestClient printed a StarletteDeprecationWarning
```

Command:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-vllm-compat \
      --kernel-launcher persistent-moe-dispatch-combine \
      --require-cuda --prompt hello --max-new-tokens 1 \
      --device 0 --arch compute_90'
```

Result:

```text
exit status: 0
status: passed
comparison_baseline: vllm-openai-compatible-deepseek
checked fixtures:
  completions
  chat_completions
  stream_completions
  stream_chat_completions
```

Each fixture recorded:

```text
HTTP status: 200
pto_status: passed
pto_launch_count: 1
launch_kind: persistent-moe-dispatch-combine
phase: prefill
dag_shape: graph_descriptor_moe_dispatch_combine
shape.n: 16
completed_count: 5
max_abs_error: 0.0
scheduler_error_summary: {count: 0, code: 0, task_id: 0}
gluon_expert_bridge.kernel_name: moe_expert_affine_f32
gluon_expert_bridge.source_sha256:
  7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f
task_body_digest.func_id: 12
task_body_digest.source_sha256:
  7cd6c62b29a6774cef62e1f00f0bbf6c106d62c82e1e10e3c571e80a5e62eb4f
device_scheduler_errors: {count: 0, code: 0, task_id: 0}
fanin_remaining: [0, 0, 0, 0, 0]
```

Route-specific compatibility summaries:

```text
completions:
  route: /v1/completions
  object: text_completion
  text_present: true
  finish_reason: length
  usage_keys: []

chat_completions:
  route: /v1/chat/completions
  object: chat.completion
  message_role: assistant
  message_content: true
  finish_reason: length
  usage_keys: []

stream_completions:
  route: /v1/completions
  stream: true
  event_count: 2
  chunk_count: 1
  done_seen: true
  assembled_text: N
  finish_reason: length

stream_chat_completions:
  route: /v1/chat/completions
  stream: true
  event_count: 2
  chunk_count: 1
  done_seen: true
  assembled_assistant_text: N
  finish_reason: length
```

The cloned source still omits non-streaming usage fields, so the compatibility
summary records `usage: false` / `usage_keys: []` rather than synthesizing
usage.

This is selected-launcher source-route contract evidence for the synthetic
adapter. It is not DeepSeek-V4-Flash serving, not real model loading, not
tokenizer semantics, not vLLM plugin integration, not throughput or latency
evidence, not distributed expert parallelism, and not production serving
readiness.

## H200 Streaming Completion Evidence

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source-stream --require-cuda \
      --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90'
```

Result:

```text
server: pypto-serving-source
route: /v1/completions
stream: true
status: passed
status_code: 200
event_count: 3
chunk_count: 2
done_seen: true
assembled_text: NV
finish_reason: length
pto_status: passed
pto_token_ids: [1, 2]
pto_launch_count: 2
```

## H200 Streaming Chat Evidence

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --pypto-serving-source-chat-stream --require-cuda \
      --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90'
```

Result:

```text
server: pypto-serving-source
route: /v1/chat/completions
stream: true
status: passed
status_code: 200
event_count: 3
chunk_count: 2
done_seen: true
assistant_deltas: [N, V]
assembled_assistant_text: NV
finish_reason: length
pto_status: passed
pto_token_ids: [1, 2]
pto_launch_count: 2
```

## Interpretation

Together, these runs prove the actual `pypto-serving`
`create_serving_app`/`ServingServer` completion and chat routes can be driven
by a simpler-nv adapter on H200, including CUDA seed launches behind the
request path. They are stronger than the local in-repo FastAPI lookalike
because they import and execute the cloned source server routes. The chat
evidence uses the bounded non-streaming OpenAI-compatible message shape.
The streaming fixtures additionally prove the cloned source SSE routes emit
token deltas and terminal `[DONE]` for the synthetic adapter on H200.
The generated Gluon launcher selection now has a bounded H200 source-route
pass through the same cloned completion route, with generated
`moe_expert_affine_f32` metadata and numerical correctness evidence.

This is not DeepSeek-V4-Flash correctness. It is not vLLM plugin evidence. It
is not real model loading, tokenizer semantics, throughput, latency, or
multi-node evidence. It is not fused MoE dispatch/combine serving readiness.
