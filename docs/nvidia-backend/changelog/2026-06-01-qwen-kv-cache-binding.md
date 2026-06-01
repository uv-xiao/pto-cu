# 2026-06-01 Qwen KV-Cache Binding

## Code And Data Changed

- Added `examples/cuda/qwen_kv_cache_binding.py` plus
  `examples/cuda/qwen_kv_cache_binding_impl/`.
- The artifact derives KV-cache sizes from the Qwen serving lifecycle plan,
  splits each planned cache into key and value buffers, and binds them to
  persistent DAG `c` and `d` fields.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Added `--cuda-live` to allocate all planned KV-cache buffers on CUDA,
  bind their device pointers to persistent DAG `c` and `d`, and close the
  owner after recording freed pointer counts.
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
`total_byte_count=5851054080` across the MPK and VDCores serving batch
ladders.

Additional CUDA-live command:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_kv_cache_binding.py \
  --cuda-live --device 0 --output-json \
  tmp/cuda-backend/pto-serving-kv-cache-live-2026-06-02/\
qwen-kv-cache-binding.json
```

Result: `mode=cuda_live`, `pointer_count=20`,
`freed_pointer_count=20`, and `total_byte_count=5851054080`.

## Remaining Gaps

- Generate Qwen attention kernels that consume persistent DAG `c` and `d`
  together with token fields, weight `tensor_args`, and token-position state.
