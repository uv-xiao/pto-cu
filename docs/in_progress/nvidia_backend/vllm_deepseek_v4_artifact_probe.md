# vLLM DeepSeek V4 Artifact Probe

This note records a bounded local artifact readiness probe for
`deepseek-ai/DeepSeek-V4-Flash`. The probe inspects files needed before a later
model-load gate and composes the existing vLLM DeepSeek V4 import/config
probes. It does not load model weights.

## Probe Surface

Tracked files:

- `examples/cuda/vllm_deepseek_v4_artifact_probe.py`
- `tests/ut/py/test_vllm_deepseek_v4_artifact_probe.py`

The probe reports structured JSON for:

- local artifact directory status;
- `config.json`, tokenizer file, tokenizer config, and weight index presence;
- indexed, present, and missing safetensors shard counts;
- selected safe fields from `config.json`;
- selected safe fields from `tokenizer_config.json`;
- existing vLLM DeepSeek V4 import readiness;
- existing vLLM DeepSeek V4 synthetic config readiness.

Missing local artifacts return `status: skipped` by default. Use
`--require-artifacts` when missing or incomplete artifacts should fail the
command. Missing installed vLLM returns a structured skip by default. Use
`--require-vllm` when installed vLLM is required.

## Local Commands

Create repo-relative symlinks under `tmp/` when the large artifacts live
outside this checkout:

```bash
mkdir -p tmp/model-artifacts/deepseek-ai
ln -s <local-artifact-dir> \
  tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
```

Run the probe without attempting model load:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
```

Require local artifact completeness:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts
```

Require both local artifacts and installed vLLM readiness:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm
```

## Local Verification

Focused tests:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_vllm_deepseek_v4_artifact_probe.py -q
```

Diff hygiene:

```bash
git diff origin/main...HEAD --check
```

## Interpretation

A passing artifact probe means the local gitignored artifact directory has the
required config/tokenizer/index files and every shard named by
`model.safetensors.index.json`. A passing vLLM portion means the existing
weight-free import/config probes also passed in the active Python environment.

This is a pre-load readiness gate. It intentionally avoids creating a vLLM
engine, allocating model weights, starting a server, or running inference.

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not H200 serving evidence.
- This is not vLLM server health evidence.
- This is not correct-text, tokenizer semantic, long-context, latency,
  throughput, or production readiness evidence.
- This does not validate raw weight tensor contents beyond index membership
  and local file presence.
