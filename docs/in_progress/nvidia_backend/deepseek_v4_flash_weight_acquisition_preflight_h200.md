# DeepSeek V4 Flash Weight Acquisition Preflight

This note records the dry-run DeepSeek-V4-Flash weight acquisition preflight
gate. It is a planning gate before any real shard download, model load, vLLM
server start, or generated-text request.

## Command

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  examples/cuda/deepseek_v4_flash_weight_acquisition_preflight.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --metadata tmp/sources/model-metadata/deepseek-ai-DeepSeek-V4-Flash.json \
  --download-root tmp \
  --reserve-bytes 10737418240 \
  --capacity-multiplier 1.1 \
  --require-capacity
```

The script reads `model.safetensors.index.json` when present, optionally
combines Hugging Face metadata, and checks the selected download root free
space. It reports `can_attempt_download` and `can_attempt_model_load`
separately. `can_attempt_model_load` remains false unless the indexed manifest
is complete.

Use `--fetch-hf-metadata` only when the host has outbound access to
Hugging Face. The H200 evidence below used a small repo-relative metadata JSON
because the remote host could not reliably reach the Hugging Face API during
the run.

## Remote H200 Dry-Run Result

The remote H200 checkout had no weight artifact directory or safetensors
index. A small metadata JSON under `tmp/sources/model-metadata/` provided the
Hugging Face `usedStorage` and safetensors total values without downloading
weight shards. The selected `tmp` filesystem did not have enough free space
for the estimated remaining bytes plus reserve, so `--require-capacity`
intentionally exited nonzero:

```text
PROBE_EXIT_STATUS=3
model_id: deepseek-ai/DeepSeek-V4-Flash
artifact_dir: tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
download_root: tmp
indexed_shard_count: 0
present_shard_count: 0
missing_shard_count: 0
indexed_bytes: null
present_bytes: 0
metadata_source: file
metadata_storage_bytes: 159641337663
metadata_safetensors_total_bytes: 158069433298
estimated_required_bytes_remaining: 159641337663
filesystem_free_bytes: 39163822080
capacity_multiplier: 1.1
reserve_bytes: 10737418240
required_capacity_bytes: 186342889670
has_required_capacity: false
can_attempt_download: false
can_attempt_model_load: false
preflight_status: blocked_storage_capacity
```

The nonzero exit is the intended capacity gate. It did not proceed to shard
download because the selected root lacked the required free bytes.

## Non-Claims

- This is not serving evidence.
- This is not model-load evidence.
- This is not DeepSeek correctness evidence.
- no shard download was attempted.
- no model load was attempted.
- no vLLM server was started.
- no generated text was produced.
- Raw model shard contents, symlink targets, private absolute paths, and
  generated-text digests are not recorded.
