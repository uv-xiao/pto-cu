# Qwen First-Layer Logits Resource Execution

## Code And Data Changed

- Added `--resource-backed-task-selection first_layer_with_logits` to the Qwen
  decode-loop runner.
- Kept the default resource-backed selection as `prefix`, preserving existing
  bounded-prefix diagnostics.
- Added selector coverage so the bounded diagnostic can execute embedding,
  layer 0, final RMSNorm, and logits in one persistent-device run.

## Architecture Quality

The selector makes the bounded diagnostic match the callable sequence a
reviewer expects for one decoder-layer slice plus logits:
`qwen_embedding_lookup`, layer 0 attention and MLP tasks, `qwen_final_norm`,
and `qwen_logits`. This avoids pretending that the first ten materialized
descriptors are a complete layer-to-logits chain, because the full materialized
Qwen graph is layer-major and otherwise continues into layer 1.

The raw execution result records the selection policy in
`resource_backed_execution.repeat_policy.task_selection`, so future evidence
can distinguish prefix diagnostics from representative callable-chain
diagnostics.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_resource_backed_execution_reports_task_coverage \
  tests/ut/py/test_nvidia_qwen_graph_materialization.py::test_resource_backed_first_layer_logits_selector_keeps_final_tasks \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode mock \
  --single-context-live-session --run-resource-backed-smoke \
  --token-cuda-live --kv-cuda-live --resident-cuda-live --workspace-cuda-live \
  --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-max-tasks 10 --resource-backed-worker-blocks 10 \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --resource-backed-logits-check-policy final_step \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-full-task-coverage/runner-cache \
  --output-json tmp/cuda-backend/qwen-full-task-coverage/qwen-decode-loop-runner.json
```

Result: tests passed; the live resource-backed diagnostic completed with
`resource_backed_execution.status=pass`, `task_coverage.task_count=10`, and
`func_id_sequence=[7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108,
7109]`. Both MPK-policy and VDCores-policy workloads reported
`graph_task_count=10`, zero scheduler errors, checked logits on the final
step, and `device_token_feedback_observed`.

## Remaining Gaps

This is still a diagnostic resource-backed execution, not a paper-promotable
full-serving Qwen row. Paper readiness still requires full Qwen numerical
correctness and latency/throughput rows with `serving_coverage=full_serving`
for the MPK and VDCores policies.
