# 2026-06-01 Qwen Decode Loop Live Bridge

## Code And Data Changed

- Added `cuda_live_bridge_contract` to
  `examples/cuda/qwen_decode_loop_runner.py`.
- Updated the PTO serving preflight so the decode-loop runner check requires
  the bridge evidence, not only the dry-run owner-ordering plan.
- Captured current evidence at
  `tmp/cuda-backend/pto-serving-decode-loop-bridge-2026-06-01/`
  `qwen-decode-loop-runner.json`.

## Architecture Quality

The runner now records how token, KV-cache, output, and resident-weight
resource owners map into the repeated proxy live runner fields:
`a`, `b`, `out`, `c`, `d`, and `tensor_args`. The contract is explicitly
`diagnostic_microdecode`; it does not satisfy full Qwen serving coverage.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline \
  --output-json \
  tmp/cuda-backend/pto-serving-decode-loop-bridge-2026-06-01/\
qwen-decode-loop-runner.json
```

Result: `status=decode_loop_runner_plan_ready` and
`cuda_live_bridge_contract.status=diagnostic_bridge_ready`.

## Remaining Gaps

- Replace proxy arithmetic with numerically correct Qwen task bodies.
- Execute the full Qwen decode loop and import full-serving viewer rows.
