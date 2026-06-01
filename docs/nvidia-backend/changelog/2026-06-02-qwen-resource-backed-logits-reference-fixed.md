# 2026-06-02 Qwen Resource-Backed Logits Reference Fix

## Code And Data Changed

- Removed early `return;` paths from generated Qwen task bodies that run
  inside the persistent DAG grid-stride wrapper.
- Added a focused unit guard so Qwen task-body snippets cannot exit that
  wrapper early again.
- Imported the fixed resource-backed logits reference artifact into the
  benchmark viewer data.

Raw artifact:

```text
tmp/cuda-backend/pto-serving-resource-backed-logits-reference-fixed-2026-06-02/qwen-decode-loop-runner.json
```

## Architecture Quality

The persistent-device task-body snippets now behave like inline loop bodies.
That matches the current source generator contract, where each snippet is
wrapped by `for (i = threadIdx.x; i < task->n; i += blockDim.x)`.

The bug previously let each scheduler worker write only the first
`blockDim.x` logits elements for snippets that returned from inside the
wrapper. The fix keeps all lanes in the wrapper until their grid-stride loop
is complete.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode offline --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-repeat-runs 3 \
  --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-logits-reference-fixed-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 3 submissions, 765 completed tasks, 0 scheduler
  errors, 2,430,976 logits elements written, 65,536 checked, diagnostic
  reference status `pass`, max absolute error `0.0`.
- `vdcores_offline_decode`: 3 submissions, 765 completed tasks, 0 scheduler
  errors, 2,430,976 logits elements written, 65,536 checked, diagnostic
  reference status `pass`, max absolute error `0.0`.

## Remaining Gaps

- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Execute the full token-by-token decode loop and import full-serving PTO rows
  for `mpk_offline_decode` and `vdcores_offline_decode`.
- Keep the paper-grade results criterion blocked until MPK, VDCores, and
  serving baseline artifacts are complete.
