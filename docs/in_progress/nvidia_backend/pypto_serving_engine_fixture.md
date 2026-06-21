# pypto-serving Engine Fixture

This note records the synthetic engine fixture added to
`examples/cuda/pypto_serving_nv_shim.py`. It is a narrow bridge from the
standalone simpler-nv executor smoke toward `pypto-serving`'s `LLMEngine`
contract.

## Contract

`SyntheticPyptoServingEngine` owns the same high-level responsibilities as the
source `pypto-serving` `LLMEngine` path, but only for the synthetic model:

- `init_model(...)` registers `synthetic-simpler-nv` from
  `synthetic://simpler-nv`.
- `generate(...)` routes prompt prefill and decode through `SimplerNvExecutor`
  and `SimplerNvModelRunner`.
- `create_openai_completion(...)` shapes the generated result through the
  same synthetic `CompletionResponse` fields used by the OpenAI-compatible
  fixture.

The fixture keeps the `ModelExecutor` boundary explicit: the engine owns model
registration and request-level generation, while `SimplerNvExecutor` and
`SimplerNvModelRunner` own CUDA launch dispatch.

The CLI flag is `--engine`. It emits the engine-owned generation result:

```text
engine: SyntheticPyptoServingEngine
model_dir: synthetic://simpler-nv
text: NV
status: passed
```

When paired with `--openai-completion`, the response includes:

```text
pto_engine: SyntheticPyptoServingEngine
object: text_completion
```

## Verification

Focused local unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py \
    -q -k 'synthetic_pypto_engine or engine_completion or engine_cli'
```

Result:

```text
3 passed, 6 deselected
```

Full local shim tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_pypto_serving_nv_shim.py -q
```

Result:

```text
9 passed
```

Remote H200 engine check:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
      --engine --prompt hello --max-new-tokens 2 \
      --device 0 --arch compute_90 --require-cuda'
```

Result:

```text
engine: SyntheticPyptoServingEngine
model_id: synthetic-simpler-nv
model_dir: synthetic://simpler-nv
status: passed
text: NV
token_ids: [1, 2]
launch_count: 2
prefill cuda_seed status: pass
decode cuda_seed status: pass
runtime: host_schedule
ptx_arch: compute_90
op: add
```

## Interpretation

This fixture proves the simpler-nv serving shim now has an engine-level owner
for model registration and request generation. It is a stronger boundary than
the raw helper because the synthetic model is initialized before generation,
matching the shape of `pypto-serving`'s `LLMEngine.init_model(...)` and
`LLMEngine.generate(...)` flow.

This is not HTTP serving evidence. It is not DeepSeek-V4-Flash correctness. It
is not vLLM plugin evidence. It is not tokenizer, KV-cache, scheduler,
streaming, throughput, latency, or multi-node evidence.
