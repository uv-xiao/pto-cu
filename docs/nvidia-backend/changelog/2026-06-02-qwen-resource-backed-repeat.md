# 2026-06-02 Qwen Resource-Backed Repeat Execution

## Code And Data Changed

- Added `--resource-backed-repeat-runs` to
  `examples/cuda/qwen_decode_loop_runner.py`.
- Updated the resource-backed Qwen execution path to reuse one prepared CUDA
  callable while submitting fresh graph state repeatedly inside the same CUDA
  context.
- Imported the repeated diagnostic rows into the benchmark viewer and kept the
  PTO full-serving work item blocked.

Raw artifact:

```text
tmp/cuda-backend/pto-serving-resource-backed-repeat-2026-06-02/qwen-decode-loop-runner.json
```

## Architecture Quality

This narrows the PTO full-serving gap from one-shot resource-backed execution
to bounded repeated `run_prepared` execution. The runtime path now records the
prepared-callable reuse policy, repeat count, per-repeat counters, and
aggregate scheduler completion counts without changing the default one-shot
behavior.

The evidence is still intentionally diagnostic. The task bodies are not yet
numerically correct Qwen kernels, and the runner does not yet perform
token-by-token full-serving decode with Qwen correctness.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py \
  --mode mock --single-context-live-session \
  --run-resource-backed-smoke --resource-backed-repeat-runs 3 \
  --token-cuda-live --kv-cuda-live --resident-cuda-live \
  --workspace-cuda-live --device 0 --arch compute_80 \
  --output-json tmp/cuda-backend/pto-serving-resource-backed-repeat-2026-06-02/qwen-decode-loop-runner.json
```

Result:

- `mpk_offline_decode`: 3 submissions, 765 completed tasks, 0 errors.
- `vdcores_offline_decode`: 3 submissions, 765 completed tasks, 0 errors.

## Remaining Gaps

- Replace diagnostic task bodies with numerically correct Qwen kernels.
- Execute the full token-by-token decode loop and import full-serving PTO rows
  for `mpk_offline_decode` and `vdcores_offline_decode`.
