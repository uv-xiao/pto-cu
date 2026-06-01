# 2026-06-01 Qwen Safetensors Shard Status

## Code And Data Changed

- Added `examples/cuda/qwen_safetensors_fetch.py` to emit Qwen/Qwen3-8B
  safetensors shard URLs, target paths, present/missing counts, tensor counts,
  and resumable `curl -L -C -` commands.
- Wired the shard-status artifact into
  `examples/cuda/persistent_qwen_serving_scaffold.py` and
  `.agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py`.
- Added the example to `examples/cuda/manifest.json` and documented the
  default no-download mode in `examples/cuda/README.md`.
- Added viewer evidence for
  `tmp/cuda-backend/pto-serving-shards-80373a64/qwen-safetensors-shards.json`
  and refreshed the scaffold/preflight artifact references.

## Architecture Quality

The PTO Qwen serving path now separates three reviewable weight stages:
expected index/shape inventory, shard placement, and safetensors metadata
validation. The new shard-status script does not fetch 16 GB of model weights
unless `--download` is passed explicitly, so local review can inspect source
URLs and target paths before a long network operation.

The scaffold and preflight now distinguish `qwen_safetensors_shard_plan` from
`qwen_safetensors_shards_present`. That keeps the current branch honest: the
fetch plan exists, but the real Qwen shards are still absent locally.

## Evaluation Run

The TDD red test first failed because the script was absent:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_persistent_qwen_safetensors_fetch_status_is_reviewable \
  -q
```

Result:

```text
FAILED ... can't open file '.../examples/cuda/qwen_safetensors_fetch.py'
```

After implementation, the focused test passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_persistent_qwen_safetensors_fetch_status_is_reviewable \
  -q
```

Result:

```text
1 passed
```

The current real-Qwen artifact reports five missing shards and zero present
shards:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_safetensors_fetch.py \
  --output-json \
  tmp/cuda-backend/pto-serving-shards-80373a64/qwen-safetensors-shards.json
```

## Remaining Gaps

- The real Qwen/Qwen3-8B safetensors shards still need to be downloaded or
  placed under `tmp/sources/qwen3-8b-safetensors/`.
- After all shards are present, rerun `qwen_safetensors_metadata.py` to validate
  actual shape/dtype metadata.
- CUDA weight binding, token-ID binding, KV-cache allocation/binding, Qwen
  kernel generation, decode-loop execution, and viewer-result import remain
  unimplemented.
