# 2026-06-01 Qwen Proxy Decode Loop Live

## Code And Data Changed

- Extended `examples/cuda/qwen_persistent_microdecode_live.py` with
  `--repeat-runs`.
- Extended the microdecode live plan with repeated decode-loop metadata,
  per-iteration expected values, and prepared-callable reuse evidence.
- Added raw artifact evidence under
  `tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/`.

## Architecture Quality

The live proxy now exercises the host-side decode-loop shape more closely:
one persistent-device callable is prepared once, then submitted three times
with fan-in, queue flags, and counters reset between runs while mutable KV
buffers stay resident and carry state forward. This remains controlled proxy
arithmetic, not full Qwen model execution.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_persistent_microdecode_live.py \
  --device 0 \
  --arch compute_80 \
  --repeat-runs 3 \
  --output-json tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/qwen-microdecode-loop.json
```

Result: `status=pass` on A100, `max_abs_error=0.0`,
`total_completed_count=9`, and `total_error_count=0`.

## Remaining Gaps

- Replace proxy task bodies with numerically correct Qwen kernels.
- Execute the full Qwen decode loop through `cuda/persistent_device`.
- Import full-serving PTO rows into the benchmark viewer.

