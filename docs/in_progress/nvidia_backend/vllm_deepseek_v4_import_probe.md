# vLLM DeepSeek V4 Import Probe

This note records a weight-free import readiness probe for vLLM DeepSeek V4.
The probe does not initialize a model, load weights, start a vLLM server, or
produce serving output.

## Probe Surface

Tracked files:

- `examples/cuda/vllm_deepseek_v4_import_probe.py`
- `tests/ut/py/test_vllm_deepseek_v4_import_probe.py`

The probe has two layers:

1. Static source checks against a vLLM source root.
2. Installed-package imports when `vllm` is importable in the active Python
   environment.

The source contract checks:

- `DeepseekV4ForCausalLM`
- `DeepseekV4Tokenizer`
- `DeepseekV4Config`
- `DeepseekV4FP8Config`

The installed import contract checks:

- `vllm.models.deepseek_v4:DeepseekV4ForCausalLM`
- `vllm.tokenizers.deepseek_v4:DeepseekV4Tokenizer`
- `vllm.transformers_utils.configs.deepseek_v4:DeepseekV4Config`
- `vllm.models.deepseek_v4.quant_config:DeepseekV4FP8Config`

Use `--require-vllm` when missing vLLM should fail the command instead of
returning a structured skip.

## Local Verification

Focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
    tests/ut/py/test_vllm_deepseek_v4_import_probe.py -q
```

The tests use temporary vLLM source fixtures. They do not depend on source
clones under `tmp/`.

Direct local probe:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_import_probe.py
```

In a local environment without installed vLLM, the command returns a structured
skip. If the default vLLM source root is absent, the JSON reports
`source_status: incomplete` with repo-relative missing file paths.

## Interpretation

This is a source-contract and import-readiness gate. A passing installed-vLLM
run with `--require-vllm` means only that the DeepSeek V4 model, tokenizer,
config, and FP8 config symbols are importable without loading weights.

## Non-Claims

- This is not DeepSeek-V4-Flash correctness.
- This is not H200 model-load evidence.
- This is not serving success.
- This is not vLLM plugin evidence.
- This is not a tokenizer, model-forward, long-context, throughput, latency,
  or correct-output test.
