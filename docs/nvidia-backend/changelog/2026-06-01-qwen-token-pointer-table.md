# 2026-06-01 Qwen Token Pointer Table

## Code And Data Changed

- Added `examples/cuda/qwen_token_pointer_table.py` plus focused helper modules
  under `examples/cuda/qwen_token_pointer_table_impl/`.
- The artifact owns Qwen `input_ids`, `attention_mask`, and `output_ids`
  device pointers while persistent decode arguments are materialized.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Captured the current lifecycle evidence at
  `tmp/cuda-backend/pto-serving-token-pointers-2026-06-01/`
  `qwen-token-pointer-table.json`.

## Architecture Quality

The token pointer-table owner makes the decode-loop lifetime boundary explicit:
token device pointers are valid while `PtoCudaPersistentDagTask::a`, `b`, and
`out` are materialized, then the owner closes and records freed pointer count.
The implementation is split into short modules for common helpers, pointer
table construction, and lifecycle materialization.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_token_pointer_table.py \
  --mode offline \
  --output-json \
  tmp/cuda-backend/pto-serving-token-pointers-2026-06-01/\
qwen-token-pointer-table.json
```

Result: `status=token_pointer_table_lifecycle_ready`,
`pointer_count=6`, `freed_pointer_count=6`, and nested decode args report
`status=persistent_decode_args_ready`.

## Remaining Gaps

- Invoke the owner in `cuda_live` mode from the decode-loop runner.
- Implement Qwen kernels that consume token fields, weight `tensor_args`, and
  KV-cache pointers, then execute and import full-serving viewer rows.
