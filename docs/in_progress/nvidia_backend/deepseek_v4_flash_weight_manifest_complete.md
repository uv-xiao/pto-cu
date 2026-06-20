# DeepSeek V4 Flash Weight Manifest Complete

This note records completed local weight-shard presence for
`deepseek-ai/DeepSeek-V4-Flash`. It is a storage readiness gate only.

## Command

The run used gitignored repo-relative symlinks under `tmp/` for both the model
artifact directory and the Hugging Face metadata JSON:

```bash
PYTHONPATH=$PWD:$PWD/python python3 \
  examples/cuda/deepseek_v4_flash_weight_manifest.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --metadata tmp/sources/model-metadata/deepseek-ai-DeepSeek-V4-Flash.json \
  --require-complete
```

The command exited with status `0`.

## Manifest Fields

```text
status: complete
model_id: deepseek-ai/DeepSeek-V4-Flash
indexed_tensors: 69187
indexed_shards: 46
present_shards: 46
missing_shards: 0
present_bytes: 159617149040
index_total_size: 159609485896
metadata_used_storage: 159641337663
metadata_safetensors_total: 158069433298
```

The manifest also reported `missing_examples: []` and
`non_claim: not serving evidence`.

## Interpretation

This proves that every shard named by `model.safetensors.index.json` is present
in the local gitignored artifact directory visible to the manifest command.
It also records the metadata totals that will be useful when preparing a model
load gate.

## Non-Claims

- This is not model-load evidence.
- This is not H200 serving evidence.
- This is not vLLM or pypto-serving integration evidence.
- This is not correct-text, long-context, latency, throughput, or production
  readiness evidence.
- Tokenizer files, Hugging Face metadata JSON, and raw model shards remain
  uncommitted under `tmp/`.

## Next Gate

The next gate is an explicit model-load or serving probe with its own command,
environment, hardware, and non-claims. This completed weight manifest should be
treated only as an input-readiness precondition for that later gate.
