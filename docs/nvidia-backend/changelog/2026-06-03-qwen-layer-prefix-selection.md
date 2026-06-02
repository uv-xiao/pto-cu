# 2026-06-03 Qwen Layer-Prefix Selection

## Code And Data Changed

- Added `--resource-backed-task-selection layer_prefix_with_logits` to the
  Qwen decode-loop runner.
- Added `--resource-backed-layer-count` to select a complete prefix of decoder
  layers while still including embedding, final RMSNorm, and logits.
- Recorded the selected layer count in resource-backed execution metadata.

## Architecture Quality

The resource-backed runner can now scale Qwen diagnostics by complete layer
prefixes instead of arbitrary task-count prefixes. This gives evaluation a
clean path from first-layer evidence toward the full 36-layer persistent DAG
without changing generated Qwen task bodies or adding separate examples.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py \
  -q -k 'resource_backed_layer_prefix_selector or resource_backed_first_layer_logits_selector or resource_backed_execution_result'
```

Result: passed.

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline \
  --single-context-live-session \
  --run-resource-backed-smoke \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 2 \
  --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 16 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols 512 \
  --resource-backed-logits-active-cols 512 \
  --device 0 \
  --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-layer-prefix-two-2026-06-03/qwen-decode-loop-runner.json
```

Result: passed. The raw artifact records `task_selection =
layer_prefix_with_logits`, `layer_count = 2`, `task_count = 17`, scheduler
`completed_count = 17`, and `error_count = 0`.

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python examples/cuda/qwen_decode_loop_runner.py \
  --mode offline \
  --single-context-live-session \
  --run-resource-backed-smoke \
  --resource-backed-task-selection layer_prefix_with_logits \
  --resource-backed-layer-count 4 \
  --resource-backed-decode-steps 1 \
  --resource-backed-worker-blocks 32 \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-projection-active-cols 512 \
  --resource-backed-logits-active-cols 512 \
  --device 0 \
  --arch compute_80 \
  --output-json tmp/cuda-backend/qwen-layer-prefix-four-2026-06-03/qwen-decode-loop-runner.json
```

Result: passed for both `mpk_offline_decode` and `vdcores_offline_decode`.
Each workload records `layer_count = 4`, `task_count = 31`, scheduler
`completed_count = 31`, `error_count = 0`, one checked logits step, sampled
512-column logits, diagnostic logits reference `status = pass`, 244 checked
elements, and max absolute error `4.67e-06`.

The same command shape was then scaled to eight complete layers with
`--resource-backed-layer-count 8`, `--resource-backed-worker-blocks 64`, and
artifact
`tmp/cuda-backend/qwen-layer-prefix-eight-2026-06-03/qwen-decode-loop-runner.json`.
Result: passed for both `mpk_offline_decode` and `vdcores_offline_decode`.
Each workload records `task_count = 59`, scheduler `completed_count = 59`,
`error_count = 0`, one checked logits step, 512 sampled logits, diagnostic
logits reference `status = pass`, 244 checked elements, and max absolute error
`3.4e-06`.

The same bounded diagnostic was then scaled to sixteen complete layers with
`--resource-backed-layer-count 16`, `--resource-backed-worker-blocks 96`, and
artifact
`tmp/cuda-backend/qwen-layer-prefix-sixteen-2026-06-03/qwen-decode-loop-runner.json`.
Result: passed for both `mpk_offline_decode` and `vdcores_offline_decode`.
Each workload records `task_count = 115`, scheduler `completed_count = 115`,
`error_count = 0`, one checked logits step, 512 sampled logits, diagnostic
logits reference `status = pass`, 244 checked elements, and max absolute error
`8.15e-06`.

The same bounded diagnostic was then scaled to the full 36-layer descriptor
set with `--resource-backed-layer-count 36`, `--resource-backed-worker-blocks
128`, and artifact
`tmp/cuda-backend/qwen-layer-prefix-thirtysix-2026-06-03/qwen-decode-loop-runner.json`.
Result: passed for both `mpk_offline_decode` and `vdcores_offline_decode`.
Each workload records `task_count = 255`, scheduler `completed_count = 255`,
`error_count = 0`, one checked logits step, 512 sampled logits, diagnostic
logits reference `status = pass`, 244 checked elements, and max absolute error
`6.81e-06`.

The full 36-layer MPK-policy diagnostic was then rerun with full projection
and logits columns using `--resource-backed-projection-active-cols full` and
`--resource-backed-logits-active-cols full`. The artifact is
`tmp/cuda-backend/qwen-full-columns-thirtysix-mpk-2026-06-03/qwen-decode-loop-runner.json`.
Result: passed for `mpk_offline_decode`. It records `task_count = 255`,
scheduler `completed_count = 255`, `error_count = 0`, 144 full projection
field overrides, full-vocab logits coverage with 2,430,976 written and checked
elements, diagnostic logits reference `status = pass`, 3,904 checked elements
across 16 rows, and max absolute error `1.543e-05`.

## Remaining Gaps

The selector is implemented and live-smoked for the full 36-layer descriptor
set on both serving policies. Full-column execution is proven for the
MPK-policy one-step diagnostic, but full Qwen numerical correctness still
requires the same full-column run for VDCores plus policy-length MPK and
VDCores decode, not only one diagnostic decode step.
