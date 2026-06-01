# 2026-06-01 Qwen Proxy Live Execution

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_proxy_live.py`.
- Added `examples/cuda/qwen_persistent_proxy_live_impl/` with separate plan,
  runtime-binding, and runner modules.
- Added CUDA example manifest and README coverage for the live proxy command.
- Added raw artifact evidence under
  `tmp/cuda-backend/pto-serving-proxy-live-2026-06-01/`.

## Architecture Quality

The live proxy keeps the full Qwen serving claim narrow. It compiles only the
controlled `qwen_attention_qkv` generated task body, prepares it as a
`cuda/persistent_device` callable, runs one persistent DAG task, and copies
back `out`, `c`, and `d`. The artifact still records that full Qwen kernel
correctness and full decode-loop execution are missing.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_proxy_live.py \
  --device 0 \
  --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-proxy-live-2026-06-01/qwen-proxy-live.json
```

Result: `status=pass` on A100, `max_abs_error=0.0`,
`completed_count=1`, and `error_count=0`.

## Remaining Gaps

- Replace proxy task bodies with numerically correct Qwen kernels.
- Execute the full Qwen decode loop through `cuda/persistent_device`.
- Import full-serving PTO rows into the benchmark viewer.

