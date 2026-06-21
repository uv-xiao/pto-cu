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
- CLI flag: `--pypto-serving-source`
- CLI flag: `--pypto-serving-source-chat`

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

## Local Verification

Focused source-contract tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
    -q -k 'pypto_serving_source'
```

Result:

```text
5 passed, 18 deselected
```

Full shim tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
23 passed
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

## H200 Evidence

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

## Interpretation

This proves the actual `pypto-serving` `create_serving_app`/`ServingServer`
contract can be driven by a simpler-nv adapter on H200, including a CUDA seed
launch behind the request path. It is stronger than the local in-repo FastAPI
lookalike because it imports and executes the cloned source server route. The
current source-contract evidence covers both `/v1/completions` and
`/v1/chat/completions`; the chat evidence uses the bounded non-streaming
OpenAI-compatible message shape.

This is not DeepSeek-V4-Flash correctness. It is not vLLM plugin evidence. It
is not real model loading, tokenizer semantics, streaming, throughput,
latency, or multi-node evidence.
