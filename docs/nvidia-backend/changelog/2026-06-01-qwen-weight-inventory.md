# 2026-06-01 Qwen Weight Inventory

## Code And Data Changed

- Added `examples/cuda/qwen_weight_inventory.py`, an executable safetensors
  index inventory artifact for PTO persistent-device Qwen/Qwen3-8B serving.
- Wired weight inventory into `persistent_qwen_serving_scaffold.py` and
  `pto_serving_preflight.py`.
- Added the weight-inventory example to `examples/cuda/manifest.json` and
  `examples/cuda/README.md`.
- Refreshed paper-readiness data so the LLM-serving PTO work item points at
  the weight-inventory artifact alongside the lifecycle, tokenizer, scaffold,
  and preflight artifacts.

## Architecture Quality

The PTO full-serving blocker now separates safetensors index inventory from
actual tensor loading. The weight-inventory artifact records Qwen/Qwen3-8B
safetensors shard count, tensor count, total size, and binding groups for
embedding, attention, attention norms, MLP, and norm/logits tensors.

This remains partial: tensors are not opened from safetensors shards, shapes
and dtypes are not validated, and weights are not copied or bound into CUDA
device memory.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_weight_inventory.py \
    --output-json tmp/cuda-backend/pto-serving-weights-edbd4390/qwen-weight-inventory.json
```

Result: `status=partial_inventory`, `tensor_count=399`, `shard_count=5`, and
`total_size_bytes=16381470720` from
`tmp/sources/qwen3-8b-model-safetensors-index-d117af2f.json`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py \
    --output tmp/cuda-backend/pto-serving-preflight-edbd4390/pto-serving-preflight.json
```

Result: `status=partial`, with `qwen_weight_inventory=pass` and full
Qwen/Qwen3-8B serving rows still absent.

## Remaining Gaps

- Open safetensors shards and validate tensor shapes and dtypes.
- Bind Qwen weight tensors into CUDA device memory and persistent-device task
  arguments.
- Implement runtime token-ID binding, CUDA KV-cache allocation/binding,
  generated Qwen kernel bodies, decode-loop execution, and viewer-result
  import.
