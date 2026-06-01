# 2026-06-02 Qwen Resource-Backed Logits Reference

## Code And Data Changed

- Added a bounded diagnostic reference check for the resource-backed
  `qwen_logits` task.
- Copied the sampled logits prefix, final hidden input, and first four
  diagnostic `lm_head` float slots back from device memory.
- Updated the viewer importer so failed diagnostic reference checks mark the
  result row as `correctness=fail`.

Raw artifact:

```text
tmp/cuda-backend/pto-serving-resource-backed-logits-reference-2026-06-02/qwen-decode-loop-runner.json
```

## Architecture Quality

This makes the diagnostic output path stricter. The runtime can no longer
present a scheduler-clean, full-buffer write as a correctness pass when the
sampled diagnostic formula disagrees with device output.

The failure is expected to keep the PTO full-serving row blocked. It is useful
because it gives the next worker a concrete numeric mismatch to debug instead
of a vague "not full Qwen" gap.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-repeat-runs 3 \
  --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-logits-reference-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 3 submissions, 765 completed tasks, 0 scheduler
  errors, 65,536 diagnostic logits elements checked, reference status `fail`,
  max absolute error `0.00148231`.
- `vdcores_offline_decode`: 3 submissions, 765 completed tasks, 0 scheduler
  errors, 65,536 diagnostic logits elements checked, reference status `fail`,
  max absolute error `0.00148231`.

## Remaining Gaps

- Debug the diagnostic `qwen_logits` formula mismatch before promoting the
  resource-backed output path.
- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Execute the full token-by-token decode loop and import full-serving PTO rows
  for `mpk_offline_decode` and `vdcores_offline_decode`.
