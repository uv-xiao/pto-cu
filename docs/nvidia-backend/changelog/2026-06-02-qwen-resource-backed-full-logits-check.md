# 2026-06-02 Qwen Resource-Backed Full Logits Check

## Code And Data Changed

- Changed the resource-backed Qwen diagnostic readback from a bounded prefix
  to the full written logits buffer.
- Compared every written diagnostic logits element against the current
  resource-backed formula.
- Imported the full-check artifact into the benchmark viewer data.

Raw artifact:

```text
tmp/cuda-backend/pto-serving-resource-backed-full-logits-check-2026-06-02/qwen-decode-loop-runner.json
```

## Architecture Quality

This strengthens the diagnostic correctness contract for the PTO
`cuda/persistent_device` resource-backed path. The previous artifact proved
that the full logits buffer was written but checked only a bounded prefix. The
new artifact proves the full written diagnostic logits buffer for both
serving policies.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-repeat-runs 3 \
  --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-full-logits-check-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 3 submissions, 765 completed tasks, 0 scheduler
  errors, 2,430,976 logits elements written, 2,430,976 checked, diagnostic
  reference status `pass`, max absolute error `0.0`.
- `vdcores_offline_decode`: 3 submissions, 765 completed tasks, 0 scheduler
  errors, 2,430,976 logits elements written, 2,430,976 checked, diagnostic
  reference status `pass`, max absolute error `0.0`.

## Remaining Gaps

- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Execute the full token-by-token decode loop and import full-serving PTO rows
  for `mpk_offline_decode` and `vdcores_offline_decode`.
- Keep the paper-grade results criterion blocked until MPK, VDCores, and
  serving baseline artifacts are complete.
