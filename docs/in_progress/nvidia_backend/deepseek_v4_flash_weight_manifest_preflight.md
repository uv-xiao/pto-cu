# DeepSeek V4 Flash Weight Manifest Preflight

This note records the local DeepSeek-V4-Flash weight acquisition preflight
surface. It is a manifest and storage-capacity gate only. It does not load
model weights, initialize vLLM, start a server, run inference, validate output
text, or exercise simpler-nv integration.

## Command

The preflight command checks the gitignored artifact path and a selected
gitignored storage directory:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/deepseek_v4_flash_weight_manifest.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --metadata tmp/sources/model-metadata/deepseek-ai-DeepSeek-V4-Flash.json \
  --storage-dir tmp
```

Use `--require-complete` when a caller needs all indexed shards present. That
gate still exits `2` when shards are incomplete or the manifest is missing.
Use `--require-preflight` when a caller needs the manifest to permit the
next model-load gate. That gate exits `3` for preflight blocks such as
insufficient storage capacity, missing artifacts, missing index data, or
incomplete shards.

`--storage-free-bytes` is available for deterministic unit tests and
reproducible dry runs. Normal operator checks should omit it so the tool reads
free bytes from the selected storage directory.

## Review-Safe Fields

The JSON output now includes:

```text
required_missing_bytes: bytes still needed from index_total_size - present_bytes
storage_dir: selected local storage directory, repo-relative when possible
storage_free_bytes: free bytes available in storage_dir
storage_required_bytes: same byte requirement used for the capacity check
storage_has_capacity: true, false, or null when the byte need is unknown
preflight_status: exact local state that blocks or permits model load
next_gate: next reviewable gate name
next_command: model-load probe command to run only after shards are present
```

The required byte count is derived from `model.safetensors.index.json`
`metadata.total_size` and the bytes already present on disk. It does not use a
hard-coded model size.

## Local Result

This worktree does not currently contain the small or complete local
DeepSeek-V4-Flash artifact subset under `tmp/model-artifacts/`. The preflight
therefore blocks before any shard acquisition or model-load attempt:

```text
status: missing
reason: artifact directory is missing
indexed_shards: 0
present_shards: 0
missing_shards: 0
present_bytes: 0
index_total_size: null
required_missing_bytes: null
storage_dir: tmp
storage_free_bytes: 32927020843008
storage_required_bytes: null
storage_has_capacity: null
preflight_status: blocked_missing_artifact_dir
next_gate: create_artifact_directory
```

Because the index file is absent in this checkout, the preflight cannot derive
the remaining shard bytes yet. Once `model.safetensors.index.json` is present,
the same command reports the missing-byte total and whether the selected
storage directory has enough free space for the remaining shards.

## Next Command

After all indexed shards are present and the manifest reports
`preflight_status: ready_for_model_load`, the next reviewable command is:

```bash
PYTHONPATH=$PWD:$PWD/python .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_model_load_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

That command must run only under an explicit GPU boundary such as
`CUDA_VISIBLE_DEVICES=<two ids>` and an external timeout. It is a later
model-load gate, not part of this preflight.

## Non-Claims

- This is not DeepSeek-V4-Flash model-load evidence.
- This is not vLLM engine initialization or server-start evidence.
- This is not inference, generated-text, tokenizer semantic, 256KB
  correctness, long-context, throughput, latency, or production evidence.
- This is not simpler-nv or pypto-serving integration evidence.
- This did not download, copy, commit, or inspect raw model shard contents.
