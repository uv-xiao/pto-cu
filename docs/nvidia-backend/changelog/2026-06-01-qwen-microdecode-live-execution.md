# 2026-06-01 Qwen Microdecode Live Execution

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_microdecode_live.py`.
- Added `examples/cuda/qwen_persistent_microdecode_live_impl/` with separate
  plan, graph descriptor, runner, and public API modules.
- Added CUDA example manifest and README coverage for the live microdecode
  proxy command.
- Added raw artifact evidence under
  `tmp/cuda-backend/pto-serving-microdecode-live-2026-06-01/`.

## Architecture Quality

The microdecode proxy exercises persistent-device dependency release across
three generated Qwen-shaped task bodies:
`qwen_attention_qkv -> qwen_attention_o -> qwen_logits`. It validates mutable
KV writeback, intermediate task outputs, final logits copy-back, and
device-side scheduler counters without claiming full Qwen model correctness.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_microdecode_live.py \
  --device 0 \
  --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-microdecode-live-2026-06-01/qwen-microdecode-live.json
```

Result: `status=pass` on A100, `max_abs_error=0.0`,
`completed_count=3`, and `error_count=0`.

## Remaining Gaps

- Replace proxy task bodies with numerically correct Qwen kernels.
- Execute the full Qwen decode loop through `cuda/persistent_device`.
- Import full-serving PTO rows into the benchmark viewer.

