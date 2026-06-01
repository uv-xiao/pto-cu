# 2026-06-01 Qwen Safetensors Metadata Probe

## Code And Data Changed

- Added `examples/cuda/qwen_safetensors_metadata.py`, a standard-library
  safetensors header probe for PTO persistent-device Qwen/Qwen3-8B serving.
- Wired the probe into `persistent_qwen_serving_scaffold.py` and
  `pto_serving_preflight.py` as a separate reviewable contract from the
  expected weight-shape inventory.
- Added the probe to `examples/cuda/manifest.json` and `examples/cuda/README.md`.
- Refreshed the benchmark-viewer matrix, readiness audit, and work queue to
  point at `tmp/cuda-backend/pto-serving-safetensors-ff252c1f/`.

## Architecture Quality

The PTO full-serving blocker now separates three weight-loader states:

- safetensors index and planned binding inventory;
- config-derived expected tensor shape/dtype contract;
- actual safetensors shard-header parsing and metadata comparison.

The real Qwen/Qwen3-8B probe currently reports `shards_missing` because the
five model shards are not present under `tmp/sources/qwen3-8b-safetensors/`.
That keeps the evidence honest while making the next required action concrete:
place or download the shards, then rerun the same probe to validate actual
headers before CUDA allocation and binding.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_safetensors_metadata.py \
    --weight-inventory-json tmp/cuda-backend/pto-serving-weights-e06636e9/qwen-weight-inventory.json \
    --output-json tmp/cuda-backend/pto-serving-safetensors-ff252c1f/qwen-safetensors-metadata.json
```

Result: `status=shards_missing`, `expected_shard_count=5`,
`opened_shard_count=0`, `missing_shard_count=5`,
`expected_tensor_count=399`, and `mismatch_count=0`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_qwen_serving_scaffold.py \
    --output-json tmp/cuda-backend/pto-serving-scaffold-ff252c1f/qwen-serving-scaffold.json
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py \
    --output tmp/cuda-backend/pto-serving-preflight-ff252c1f/pto-serving-preflight.json
```

Result: `qwen_safetensors_metadata_probe=pass` and
`qwen_actual_safetensors_metadata=fail`, so full Qwen/Qwen3-8B serving rows
remain blocked.

## Remaining Gaps

- Place or download the five Qwen/Qwen3-8B safetensors shards under
  `tmp/sources/qwen3-8b-safetensors/`.
- Validate actual safetensors header metadata against the expected contract.
- Bind Qwen weight tensors into CUDA device memory and persistent-device task
  arguments.
- Implement runtime token-ID binding, CUDA KV-cache allocation/binding,
  generated Qwen kernel bodies, decode-loop execution, and viewer-result
  import.
