# vLLM DeepSeek V4 Config Probe

This note records a weight-free DeepSeek V4 configuration readiness probe. The
probe does not initialize a model, load weights, read tokenizer files, start a
vLLM server, or produce serving output.

## Probe Surface

Tracked files:

- `examples/cuda/vllm_deepseek_v4_config_probe.py`
- `tests/ut/py/test_vllm_deepseek_v4_config_probe.py`

The probe has two layers:

1. Static source checks against a vLLM source root.
2. Synthetic config construction when `vllm` is importable.

The source contract checks:

- `DeepseekV4Config`
- default `max_position_embeddings: 1048576`
- `DeepseekV4FP8Config`
- quantization method `deepseek_v4_fp8`
- `expert_dtype`

The installed config contract constructs a synthetic `DeepseekV4Config` with
`max_position_embeddings`, `expert_dtype`, and quantization metadata. It does
not read model shards or tokenizer artifacts.

Use `--require-vllm` when missing vLLM should fail the command instead of
returning a structured skip.

## Local Verification

Focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_vllm_deepseek_v4_config_probe.py -q
```

The tests use temporary vLLM source fixtures and monkeypatched import modules.
They do not depend on source clones under `tmp/`.

Direct local probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_config_probe.py \
    --max-position-embeddings 262144
```

In a local environment without installed vLLM, the command returns a structured
skip. If the default vLLM source root is absent, the JSON reports
`source_status: incomplete` with repo-relative missing file paths.

## Interpretation

This is a source-contract and synthetic-config gate. A passing installed-vLLM
run with `--require-vllm` means only that a synthetic DeepSeek V4 config can be
constructed and mapped to the DeepSeek V4 FP8 quantization override without
loading weights.

## Non-Claims

- This is not DeepSeek-V4-Flash correctness.
- This is not H200 model-load evidence.
- This is not serving success.
- This is not tokenizer compatibility.
- This is not model-forward, long-context, throughput, latency, or
  correct-output evidence.
