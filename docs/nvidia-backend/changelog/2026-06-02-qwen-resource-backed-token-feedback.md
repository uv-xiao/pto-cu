# Qwen Resource-Backed Token Feedback

## Code And Data Changed

- Added diagnostic sampled-token feedback to bounded resource-backed Qwen
  decode steps.
- Added `qwen_decode_loop_runner_impl/decode_feedback.py` to keep token
  feedback copy/readback logic separate from CUDA DAG execution.
- Imported the token-feedback artifact into the benchmark-viewer results.
- Extended viewer result statistics with decode-feedback status and applied
  step count.

## Architecture Quality

The resource-backed path now records a narrow token lifecycle after each
bounded diagnostic decode step: the sampled top token is written into
`output_ids[step]`, copied into `input_ids[0]` for the next diagnostic step,
and read back for evidence. This still uses host-side feedback between
`run_prepared` submissions; it is not device-side sampling and does not claim
full CUDA persistent-kernel serving.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-decode-steps 2 --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-token-feedback-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 2 diagnostic decode steps, 510 completed tasks, zero
  scheduler errors, and 2 sampled-token feedback commits.
- `vdcores_offline_decode`: 2 diagnostic decode steps, 510 completed tasks,
  zero scheduler errors, and 2 sampled-token feedback commits.
- Both policies checked all 2,430,976 written diagnostic logits elements with
  zero mismatches.

## Remaining Gaps

This proves diagnostic token movement through current buffers, not full Qwen
serving. Remaining PTO gaps are numerically correct Qwen kernels, device-side
token sampling/feedback, full token-by-token decode-loop execution, and
full-serving viewer rows.
