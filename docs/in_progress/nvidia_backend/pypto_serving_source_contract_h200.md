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

`PyptoServingSourceAsyncEngineAdapter` adapts the synthetic simpler-nv engine
to the server's async `add_request(...)` contract. The adapter returns the
token-output shape expected by the actual server while preserving PTO debug
evidence outside the server response.

The tracked entry points are:

- `create_pypto_serving_source_app(...)`
- `run_pypto_serving_source_completion_fixture(...)`
- CLI flag: `--pypto-serving-source`

## Current Chat Source Limitation

The local synthetic shim now has `/v1/chat/completions` coverage through
`SyntheticPyptoServingEngine` and the in-repo FastAPI fixture. The actual
cloned source route is unchanged in this slice: in this worktree,
`tmp/sources/repos/hw-native-sys/pypto-serving is absent`, so
`/v1/chat/completions` support in the cloned `pypto-serving` source cannot be
inspected or exercised here.

Because the source checkout is unavailable, this slice records the limitation
instead of adding a source-route chat fixture. The source-contract fixture
continues to cover only `/v1/completions` until a worker with the cloned source
present verifies whether the real source server exposes `/v1/chat/completions`.

## Local Verification

Focused source-contract tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
    -q -k 'pypto_serving_source'
```

Result:

```text
2 passed, 13 deselected
```

Full shim tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
15 passed
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

## Interpretation

This proves the actual `pypto-serving` `create_serving_app`/`ServingServer`
contract can be driven by a simpler-nv adapter on H200, including a CUDA seed
launch behind the request path. It is stronger than the local in-repo FastAPI
lookalike because it imports and executes the cloned source server route.

This is not DeepSeek-V4-Flash correctness. It is not vLLM plugin evidence. It
is not real model loading, tokenizer semantics, streaming, source-route
chat-completion, throughput, latency, or multi-node evidence.
