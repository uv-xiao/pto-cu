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

## Remaining Gaps

The selector is implemented and live-smoked for two layers, but full Qwen
numerical correctness still requires running larger layer prefixes and then all
36 layers with policy-length MPK and VDCores workloads.
