# Qwen Resource-Backed Decode Steps

## Code And Data Changed

- Added `--resource-backed-decode-steps` to the Qwen decode-loop runner.
- Threaded the decode-step limit through the single-context resource-backed
  execution path.
- Split decode-step execution policy helpers into
  `qwen_decode_loop_runner_impl/resource_execution_policy.py`.
- Imported the new artifact into the benchmark-viewer result data.

## Architecture Quality

The runner can now distinguish repeated diagnostic submissions from bounded
decode-step execution. The resource-backed path records planned and executed
decode steps per serving policy while keeping the CUDA callable and owner
lifecycle unchanged: one CUDA context, one prepared persistent-device callable,
fresh graph state per submitted diagnostic step, and owner-held token, KV, and
weight pointers.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-decode-steps 2 --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-decode-steps-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 1024 planned decode steps, 2 executed diagnostic
  steps, 510 completed tasks, zero scheduler errors.
- `vdcores_offline_decode`: 64 planned decode steps, 2 executed diagnostic
  steps, 510 completed tasks, zero scheduler errors.
- Both policies checked all 2,430,976 written diagnostic logits elements with
  zero mismatches.

## Remaining Gaps

This is bounded diagnostic decode-step evidence, not full Qwen serving. The
remaining PTO gaps are numerically correct Qwen task bodies, full
token-by-token decode-loop execution, and full-serving viewer rows.
