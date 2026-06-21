# pypto-serving OpenAI Completion Fixture

This note records the synthetic OpenAI-compatible completion fixture added on
top of `examples/cuda/pypto_serving_nv_shim.py`. It is the next serving bridge
after `pypto_serving_nv_shim_local.md`: the existing synthetic simpler-nv
executor result is shaped like `pypto-serving`'s `/v1/completions` response.

## Contract

The fixture entry point is `run_synthetic_openai_completion(...)`. It wraps
`run_synthetic_serving_request(...)` and returns the response fields used by
`pypto-serving`'s server models:

- `CompletionResponse`: `id`, `object`, `created`, `model`, and `choices`.
- `CompletionChoice`: `index`, `text`, and `finish_reason`.
- usage fields: `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- PTO debug fields: `pto_backend`, `pto_status`, and `pto_launch_count`.

The CLI flag is `--openai-completion`. It emits a synthetic
`/v1/completions`-shaped response with deterministic metadata:

```text
id: cmpl-synthetic-0
object: text_completion
text: NV
pto_status: passed
```

## Verification

Focused local unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
    -q -k openai_completion
```

Result:

```text
2 passed, 4 deselected
```

Remote H200 API-shape check:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --openai-completion \
      --model synthetic-simpler-nv \
      --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90 --require-cuda'
```

Result:

```text
id: cmpl-synthetic-0
object: text_completion
model: synthetic-simpler-nv
choices[0].text: NV
choices[0].finish_reason: length
usage.prompt_tokens: 1
usage.completion_tokens: 2
usage.total_tokens: 3
pto_backend: simpler-nv
pto_status: passed
pto_launch_count: 2
```

## Interpretation

This fixture proves the synthetic simpler-nv shim can now produce a response
shape compatible with the core `/v1/completions` payload from
`pypto-serving`. It keeps the OpenAI-compatible API target visible while the
repository still lacks a checked-in FastAPI server integration.

This is not HTTP serving evidence. It is not DeepSeek-V4-Flash correctness. It
is not vLLM plugin evidence. It is not tokenizer, streaming, chat-completion,
throughput, latency, or multi-node evidence.
