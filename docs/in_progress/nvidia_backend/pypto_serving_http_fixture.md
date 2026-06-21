# pypto-serving HTTP Fixture

This note records the local FastAPI fixture added to
`examples/cuda/pypto_serving_nv_shim.py`. It is the first checked-in HTTP
boundary for the synthetic simpler-nv serving path.

## Contract

The fixture adds two entry points:

- `create_synthetic_openai_app(...)` builds an in-process FastAPI app.
- `run_synthetic_http_completion_fixture(...)` drives the app with
  `TestClient` and posts one `/v1/completions` request.

The app exposes the same basic routes as the source `pypto-serving`
`ServingServer`:

- `/health`
- `/v1/models`
- `/v1/completions`

The completion route initializes `SyntheticPyptoServingEngine`, then routes the
request through `create_openai_completion(...)`. The CLI flag is
`--http-fixture`.

## Local Verification

Focused local HTTP tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
    -q -k 'fastapi_app or http_fixture'
```

Result:

```text
2 passed, 9 deselected
```

Full local shim tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
12 passed
```

Local HTTP fixture evidence:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
    --http-fixture --prompt hello --max-new-tokens 2 \
    --device 0 --arch compute_90
```

Result:

```text
status: passed
route: /v1/completions
status_code: 200
object: text_completion
text: NV
pto_engine: SyntheticPyptoServingEngine
pto_status: skipped
```

Summary: local HTTP fixture evidence passed at the route level.

The route-level `status: passed` means the local FastAPI request succeeded.
The nested `pto_status: skipped` means the local machine did not provide a
successful CUDA seed launch. `--require-cuda` now checks this nested status and
returns non-zero when the HTTP route passes but CUDA is skipped.

## Remote H200 Dependency Check

Initial dependency check: the remote H200 HTTP fixture skipped because the
remote Python environment did not include FastAPI:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --http-fixture --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90'
```

Result:

```text
status: skipped
route: /v1/completions
reason: missing FastAPI TestClient: No module named 'fastapi'
```

Summary: Remote H200 HTTP fixture skipped; FastAPI unavailable.
Summary: initial dependency skip was expected before installing the remote
TestClient dependency set.

## H200 HTTP Fixture Evidence

The remote project virtual environment was then updated with the minimal
FastAPI `TestClient` dependency set:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    python -m pip install fastapi==0.128.0 starlette==0.50.0'
```

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    python -m pip install httpx==0.28.1'
```

After that controlled dependency install, the H200 HTTP fixture passed with
CUDA required:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --http-fixture --require-cuda \
      --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90'
```

Result:

```text
status: passed
route: /v1/completions
status_code: 200
object: text_completion
text: NV
pto_engine: SyntheticPyptoServingEngine
pto_status: passed
pto_launch_count: 2
```

## Interpretation

The serving path now has a local HTTP shell that exercises `/health`,
`/v1/models`, and `/v1/completions` against the synthetic simpler-nv engine,
plus H200 HTTP fixture evidence for `/v1/completions` with CUDA required. This
closes the first HTTP contract gap without adding a hard FastAPI import to the
base shim module.

This is not DeepSeek-V4-Flash correctness. It is not vLLM plugin evidence. It
is not real model loading, tokenizer, streaming, chat-completion, throughput,
latency, or multi-node evidence.
