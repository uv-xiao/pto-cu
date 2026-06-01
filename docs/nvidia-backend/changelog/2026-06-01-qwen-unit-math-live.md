# 2026-06-01 Qwen Unit Math Live Execution

## Code And Data Changed

- Added `examples/cuda/qwen_unit_math_live.py` plus a focused implementation
  under `examples/cuda/qwen_unit_math_live_impl/`.
- The live artifact executes RMSNorm, QKV cache writeback, SiLU/SwiGLU, and
  logits task bodies through `cuda/persistent_device`.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-unit-math-live-2026-06-01/`
  `qwen-unit-math-live.json`.

## Architecture Quality

The unit-math path now has runtime evidence rather than only source evidence.
The new DAG is still a hidden-size-4 diagnostic, so it does not claim full
Qwen decode-loop coverage.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_unit_math_live.py \
  --device 0 --arch compute_80 \
  --output-json \
  tmp/cuda-backend/pto-serving-unit-math-live-2026-06-01/\
qwen-unit-math-live.json
```

Result: `status=pass`, `completed_count=4`, `error_count=0`, and
`max_abs_error=0.0`.

## Remaining Gaps

- Execute the full Qwen decode loop.
- Import full-serving viewer rows.
