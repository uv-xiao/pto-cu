# 2026-06-01 Qwen Persistent Decode Arguments

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_decode_args.py`, which binds Qwen token
  device pointers to the persistent DAG `a`, `b`, and `out` fields.
- Kept Qwen weight pointers on `tensor_args`, so the token-buffer path does not
  consume the four-pointer weight-argument capacity.
- Wired the artifact into the Qwen serving scaffold, PTO serving preflight,
  CUDA example manifest, example README, paper-readiness matrix, and tests.
- Captured the current symbolic decode-argument evidence at
  `tmp/cuda-backend/pto-serving-decode-args-2026-06-01/`
  `qwen-persistent-decode-args.json`.

## Architecture Quality

The persistent decode-argument artifact makes the first decode-loop ABI
contract explicit: `input_ids` maps to `PtoCudaPersistentDagTask::a`,
`attention_mask` maps to `b`, and `output_ids` maps to `out`. The scalar
fields carry prompt length, batch size, decode length, and first decode
position. This keeps token buffers separate from Qwen weight `tensor_args`.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_decode_args.py \
  --output-json \
  tmp/cuda-backend/pto-serving-decode-args-2026-06-01/\
qwen-persistent-decode-args.json
```

Result: `status=persistent_decode_args_plan_ready`; both paper serving
policies have symbolic `a`, `b`, and `out` token pointer sources. The focused
fixture test also validates `status=persistent_decode_args_ready` when a live
token pointer table is supplied.

## Remaining Gaps

- Keep a live token pointer table open from the decode-loop runner.
- Implement Qwen kernels that consume `a`, `b`, `out`, weight `tensor_args`,
  and KV-cache pointers, then execute and import full-serving viewer rows.
