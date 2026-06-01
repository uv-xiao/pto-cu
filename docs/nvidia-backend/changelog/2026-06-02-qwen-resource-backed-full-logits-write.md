# 2026-06-02 Qwen Resource-Backed Full Logits Write

## Code And Data Changed

- Updated the resource-backed Qwen launch packet so the final `qwen_logits`
  task runs over the logits-buffer extent instead of the hidden-buffer extent.
- Passed the hidden-buffer extent to the diagnostic logits task through scalar
  metadata so it can write the full logits buffer without reading past the
  prior activation buffer.
- Imported the resulting viewer rows with
  `full_logits_buffer_prefix_sampled` coverage.

Raw artifact:

```text
tmp/cuda-backend/pto-serving-resource-backed-full-logits-write-2026-06-02/qwen-decode-loop-runner.json
```

## Architecture Quality

This removes the previous partial-write limitation in the diagnostic
resource-backed path. The artifact now records that all 2,430,976 logits
buffer elements are written for each serving policy while keeping review I/O
bounded to a 65,536-element prefix sample.

The task bodies remain diagnostic and are not yet numerically correct Qwen
kernels. The viewer and matrix therefore keep the PTO full-serving work item
blocked.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-repeat-runs 3 \
  --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-full-logits-write-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 3 submissions, 765 completed tasks, 0 errors,
  2,430,976 logits elements written, 65,536 prefix elements sampled, stable
  output summary.
- `vdcores_offline_decode`: 3 submissions, 765 completed tasks, 0 errors,
  2,430,976 logits elements written, 65,536 prefix elements sampled, stable
  output summary.

## Remaining Gaps

- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Validate full-vocabulary logits against a Qwen reference, not just buffer
  write coverage.
- Execute the full token-by-token decode loop and import full-serving PTO rows
  for `mpk_offline_decode` and `vdcores_offline_decode`.
