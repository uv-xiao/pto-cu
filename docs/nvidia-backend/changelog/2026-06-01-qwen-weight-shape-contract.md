# 2026-06-01 Qwen Weight Shape Contract

## Code And Data Changed

- Extended `examples/cuda/qwen_weight_inventory.py` with `--config-json` and a
  config-derived expected shape/dtype contract for Qwen/Qwen3-8B weights.
- Refreshed the PTO Qwen weight, scaffold, and preflight evidence roots to
  `tmp/cuda-backend/pto-serving-weights-e06636e9/`,
  `tmp/cuda-backend/pto-serving-scaffold-e06636e9/`, and
  `tmp/cuda-backend/pto-serving-preflight-e06636e9/`.
- Updated the benchmark-viewer matrix, paper-readiness audit, work queue,
  examples manifest, examples README, evaluation plan, and baseline survey to
  describe the expected shape contract separately from actual safetensors
  metadata validation.

## Architecture Quality

The PTO serving blocker now has a deterministic weight-shape contract derived
from the captured Qwen config instead of only shard/tensor names. For each
indexed tensor, the artifact records expected shape, dtype, and byte size. The
real Qwen/Qwen3-8B inventory covers 399 tensors and reconciles
`expected_total_size_bytes=16381470720` with the safetensors index total.

This is still not a tensor loader. The remaining loader work must open the
safetensors shards, validate actual tensor metadata against this expected
contract, allocate/copy weights on CUDA, and bind the device pointers into
persistent-device task arguments.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_weight_inventory.py \
    --output-json tmp/cuda-backend/pto-serving-weights-e06636e9/qwen-weight-inventory.json
```

Result: `status=partial_inventory`, shape contract
`status=complete_for_index`, `tensor_count=399`,
`expected_total_size_bytes=16381470720`, and `size_matches_index=true`.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_qwen_serving_scaffold.py \
    --output-json tmp/cuda-backend/pto-serving-scaffold-e06636e9/qwen-serving-scaffold.json
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py \
    --output tmp/cuda-backend/pto-serving-preflight-e06636e9/pto-serving-preflight.json
```

Result: Qwen weight loader remains `partial`; `qwen_weight_inventory=pass`;
full Qwen/Qwen3-8B serving rows remain absent.

## Remaining Gaps

- Open safetensors shards and validate actual tensor shapes and dtypes.
- Bind Qwen weight tensors into CUDA device memory and persistent-device task
  arguments.
- Implement runtime token-ID binding, CUDA KV-cache allocation/binding,
  generated Qwen kernel bodies, decode-loop execution, and viewer-result
  import.
