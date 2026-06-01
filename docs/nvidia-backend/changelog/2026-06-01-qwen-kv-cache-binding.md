# 2026-06-01 Qwen KV-Cache Binding

## Code And Data Changed

- Added `examples/cuda/qwen_kv_cache_binding.py` plus
  `examples/cuda/qwen_kv_cache_binding_impl/`.
- The artifact derives KV-cache sizes from the Qwen serving lifecycle plan,
  splits each planned cache into key and value buffers, and binds them to
  persistent DAG `c` and `d` fields.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-kv-cache-2026-06-01/`
  `qwen-kv-cache-binding.json`.

## Architecture Quality

The KV-cache binding keeps token pointers on `a`/`b`/`out`, weight pointers on
`tensor_args`, and attention KV-cache pointers on `c`/`d`. This makes the
current persistent DAG ABI pressure explicit before Qwen attention kernels are
generated.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_kv_cache_binding.py \
  --output-json \
  tmp/cuda-backend/pto-serving-kv-cache-2026-06-01/\
qwen-kv-cache-binding.json
```

Result: `status=kv_cache_lifecycle_ready`, `pointer_count=20`, and
`total_byte_count=5858476032` across the MPK and VDCores serving batch
ladders.

## Remaining Gaps

- Run a real `cuda_live` KV-cache owner from the decode-loop runner.
- Generate Qwen attention kernels that consume persistent DAG `c` and `d`
  together with token fields, weight `tensor_args`, and token-position state.
