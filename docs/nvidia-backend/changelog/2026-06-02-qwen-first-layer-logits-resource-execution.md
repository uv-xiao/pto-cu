# Qwen First-Layer Logits Resource Execution

## Code And Data Changed

- Added `--resource-backed-task-selection first_layer_with_logits` to the Qwen
  decode-loop runner.
- Kept the default resource-backed selection as `prefix`, preserving existing
  bounded-prefix diagnostics.
- Added selector coverage so the bounded diagnostic can execute embedding,
  layer 0, final RMSNorm, and logits in one persistent-device run.
- Extended the resource-backed viewer importer to preserve task-selection,
  task func-id sequence, and callable sequence metadata.
- Made the resource-backed matrix import record the raw execution's logits
  check policy instead of labeling every resource-backed artifact as
  final-step-only.
- Imported the two first-layer/logits diagnostic rows into the benchmark
  viewer, one for `mpk_offline_decode` and one for `vdcores_offline_decode`.
- Added a four-step first-layer/logits diagnostic run that checks logits and
  device-committed sampled-token feedback on every executed decode step.
- Added a sixteen-step first-layer/logits diagnostic run after the 64-step
  attempt exceeded the quick convergence window without producing an artifact.
- Added a VDCores-only 64-step first-layer/logits diagnostic run with
  final-step logits checking, giving policy-length coverage for that workload
  without committing the raw artifact.

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
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/pto_qwen_resource_backed_viewer_import.py \
  tmp/cuda-backend/qwen-full-task-coverage/qwen-decode-loop-runner.json \
  --artifact-root tmp/cuda-backend/qwen-full-task-coverage/ \
  --commit c98eabff
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_work_queue.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-decode-steps 4 \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-logits-check-policy every_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-first-layer-logits-4step/runner-cache \
  --output-json tmp/cuda-backend/qwen-first-layer-logits-4step/qwen-decode-loop-runner.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/pto_qwen_resource_backed_viewer_import.py \
  tmp/cuda-backend/qwen-first-layer-logits-4step/qwen-decode-loop-runner.json \
  --artifact-root tmp/cuda-backend/qwen-first-layer-logits-4step/ \
  --commit c552ae72
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_work_queue.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/nvidia_goal_progress.py \
  --output evaluations/nvidia/benchmark-viewer/data/goal_progress.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-decode-steps 16 \
  --resource-backed-workload mpk_offline_decode \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-logits-check-policy every_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-first-layer-logits-16step/runner-cache \
  --output-json tmp/cuda-backend/qwen-first-layer-logits-16step/qwen-decode-loop-runner.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/pto_qwen_resource_backed_viewer_import.py \
  tmp/cuda-backend/qwen-first-layer-logits-16step/qwen-decode-loop-runner.json \
  --artifact-root tmp/cuda-backend/qwen-first-layer-logits-16step/ \
  --commit 84a77216
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_work_queue.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/nvidia_goal_progress.py \
  --output evaluations/nvidia/benchmark-viewer/data/goal_progress.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  examples/cuda/qwen_decode_loop_runner.py --mode offline \
  --single-context-live-session --run-resource-backed-smoke \
  --resource-backed-task-selection first_layer_with_logits \
  --resource-backed-decode-steps 64 \
  --resource-backed-workload vdcores_offline_decode \
  --resource-backed-logits-check-policy final_step \
  --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
  --device 0 --arch compute_80 \
  --cache-root tmp/cuda-backend/qwen-first-layer-logits-vdcores-64step-final/runner-cache \
  --output-json tmp/cuda-backend/qwen-first-layer-logits-vdcores-64step-final/qwen-decode-loop-runner.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/pto_qwen_resource_backed_viewer_import.py \
  tmp/cuda-backend/qwen-first-layer-logits-vdcores-64step-final/qwen-decode-loop-runner.json \
  --artifact-root tmp/cuda-backend/qwen-first-layer-logits-vdcores-64step-final/ \
  --commit 6035348b
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_audit.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_readiness_work_queue.py \
  --output evaluations/nvidia/benchmark-viewer/data/paper_readiness_work_queue.json
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/nvidia_goal_progress.py \
  --output evaluations/nvidia/benchmark-viewer/data/goal_progress.json
```

Result: tests passed; the live resource-backed diagnostic completed with
`resource_backed_execution.status=pass`, `task_coverage.task_count=10`, and
`func_id_sequence=[7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108,
7109]`. Both MPK-policy and VDCores-policy workloads reported
`graph_task_count=10`, zero scheduler errors, checked logits on the final
step, and `device_token_feedback_observed`.

The benchmark viewer now contains two compact rows under
`tmp/cuda-backend/qwen-full-task-coverage/`. Each row records
`task_selection=first_layer_with_logits`, `task_coverage_count=10`,
`task_func_id_sequence=[7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107,
7108, 7109]`, checked logits, zero scheduler errors, and device token
feedback.

The later four-step run under
`tmp/cuda-backend/qwen-first-layer-logits-4step/` also passed for both
workloads. Each compact viewer row records `executed_decode_steps=4`,
`logits_check_policy=every_step`, `logits_checked_step_count=4`, zero
scheduler errors, `diagnostic_logits_reference_status=pass`, and
`decode_feedback_applied_step_count=4` with device-observed token feedback.

The sixteen-step run under
`tmp/cuda-backend/qwen-first-layer-logits-16step/` passed for both workloads.
The imported rows record `executed_decode_steps=16`,
`logits_checked_step_count=16`, zero scheduler errors,
`diagnostic_logits_reference_status=pass`, and
`decode_feedback_applied_step_count=16`. This is still bounded for the
MPK-policy workload and not full policy length for VDCores, but it exercises
the representative first-layer/logits chain over more decode positions than
the four-step smoke while keeping committed viewer data compact.

The VDCores-only 64-step final-check run under
`tmp/cuda-backend/qwen-first-layer-logits-vdcores-64step-final/` passed with
`decode_step_execution.status=policy_length_decode_steps_executed`,
`executed_decode_steps=64`, `logits_checked_step_count=1`,
`logits_deferred_step_count=63`, zero scheduler errors,
`diagnostic_logits_reference_status=pass`, and
`decode_feedback_applied_step_count=64`. A combined MPK-plus-VDCores 64-step
attempt exceeded the quick convergence window and produced no importable
artifact.

## Remaining Gaps

This is still a diagnostic resource-backed execution, not a paper-promotable
full-serving Qwen row. Paper readiness still requires full Qwen numerical
correctness and latency/throughput rows with `serving_coverage=full_serving`
for the MPK and VDCores policies.
