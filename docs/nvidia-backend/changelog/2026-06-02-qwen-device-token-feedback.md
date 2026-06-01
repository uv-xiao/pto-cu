# Qwen Device Token Feedback

## Code And Data Changed

- Added a generated `qwen_logits` diagnostic feedback path that writes the
  sampled token from device code into `output_ids[decode_step]` and
  `input_ids[0]`.
- Passed token-buffer pointers to the final logits task through spare
  persistent-DAG `tensor_args` slots.
- Split launch-packet helpers and the logits feedback task-body spec into
  focused modules to keep edited files review-sized.
- Imported the new device-feedback artifact into the benchmark-viewer data.

## Architecture Quality

The bounded resource-backed decode loop no longer relies on host writes to
commit the diagnostic sampled token. The host still launches each bounded
diagnostic step and reads back evidence, but the token feedback state change
itself is performed inside the persistent-device task body.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-decode-steps 2 --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-device-feedback-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 2 diagnostic decode steps, 510 completed tasks, zero
  scheduler errors, and 2 device-observed sampled-token feedback commits.
- `vdcores_offline_decode`: 2 diagnostic decode steps, 510 completed tasks,
  zero scheduler errors, and 2 device-observed sampled-token feedback commits.
- Both policies checked all 2,430,976 written diagnostic logits elements with
  zero mismatches.

## Remaining Gaps

This proves device-side diagnostic token feedback, not full Qwen serving.
Remaining PTO gaps are numerically correct Qwen kernels, full token-by-token
decode-loop execution for the serving policies, and full-serving viewer rows.
