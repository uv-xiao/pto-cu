# 2026-06-02 Qwen Bounded Full RMSNorm Diagnostic

## Code And Data Changed

- Capped `unit_math_full_rmsnorm` hidden-stage resource-backed packets to one
  4,096-element Qwen hidden vector.
- Exposed `element_limit=4096` in the full-RMSNorm numeric-mode contract.
- Imported the bounded A100 resource-backed diagnostic artifact into the
  benchmark viewer as two `diagnostic_resource_backed_qwen_dag` rows.
- Imported a VDCores-policy row that combines
  `unit_math_full_rmsnorm` with `--resource-backed-logits-active-cols full`.
- Fixed full-RMSNorm branch selection so decode-position metadata in
  `scalar_args[2]` does not demote full reductions into the external-scale
  branch.
- Imported the corrected VDCores-policy row with nonzero full-logits output.
- Imported the corrected MPK-policy row with the same full-RMSNorm and
  full-logits contract.
- Pruned superseded resource-backed viewer rows that only covered partial
  logits, scaffold-only smoke, or the stale zero-logits full-RMSNorm result.

## Architecture Quality

The generated RMSNorm body still performs a per-output-element reduction, so
using the full MPK-policy batch buffer would make the diagnostic impractical.
The new cap keeps this path runnable while preserving the review distinction
between the external-scale RMSNorm shortcut and the generated reduction
branch. The rows remain diagnostic and cannot satisfy the full-serving gate.

## Evaluation Run

- Passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_graph_materialization.py \
    -q -k full_rmsnorm
  ```
- Passed live A100 diagnostic:
  `examples/cuda/qwen_decode_loop_runner.py` with
  `--resource-backed-numeric-task-mode unit_math_full_rmsnorm`,
  `--resource-backed-decode-steps 1`, and output artifact under
  `tmp/cuda-backend/pto-serving-resource-backed-full-rmsnorm-2026-06-02/`.
- Result: both `mpk_offline_decode` and `vdcores_offline_decode` completed
  255 resource-backed tasks with zero scheduler errors and full logits-buffer
  diagnostic reference checks passing.
- Historical live A100 combined diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-full-active-logits-full-rmsnorm-vdcores-2026-06-03
  PYTHONPATH=$PWD:$PWD/python timeout 480 .venv/bin/python \
    examples/cuda/qwen_decode_loop_runner.py --mode mock \
    --single-context-live-session --run-resource-backed-smoke \
    --resource-backed-task-selection first_layer_with_logits \
    --resource-backed-workload vdcores_offline_decode \
    --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
    --resource-backed-worker-blocks 10 \
    --resource-backed-logits-check-policy final_step \
    --resource-backed-logits-active-cols full \
    --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
    --device 0 --arch compute_80 \
    --cache-root $ARTIFACT/cache \
    --output-json $ARTIFACT/qwen-decode-loop-runner.json
  ```

- Result: `vdcores_offline_decode` completed 10 resource-backed task
  functions with zero scheduler errors, `full_reduction_contract_count=3`,
  `full_logits_buffer_checked` over 2,430,976 logits elements, and diagnostic
  projection reference `max_abs_error=0.0`. The diagnostic logits were all
  zero in this mode, so this run only proved the combined contract execution
  path. Its compact viewer row was removed after the marker fix superseded it.
- Passed focused regressions after the marker fix:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_graph_materialization.py \
    -q -k 'full_rmsnorm or launch_packet_can_select_full_rmsnorm'

  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
    tests/ut/py/test_nvidia_qwen_task_body_math.py \
    -q -k generated_source_contains_qwen_unit_math_kernels
  ```

- Passed corrected live A100 diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-full-rmsnorm-marker-fix-clean-vdcores-2026-06-03
  PYTHONPATH=$PWD:$PWD/python timeout 480 .venv/bin/python \
    examples/cuda/qwen_decode_loop_runner.py --mode mock \
    --single-context-live-session --run-resource-backed-smoke \
    --resource-backed-task-selection first_layer_with_logits \
    --resource-backed-workload vdcores_offline_decode \
    --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
    --resource-backed-worker-blocks 10 \
    --resource-backed-logits-check-policy final_step \
    --resource-backed-logits-active-cols full \
    --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
    --device 0 --arch compute_80 \
    --cache-root $ARTIFACT/cache \
    --output-json $ARTIFACT/qwen-decode-loop-runner.json
  ```

- Corrected result: `vdcores_offline_decode` completed 10 resource-backed task
  functions with zero scheduler errors, `full_logits_buffer_checked` over
  2,430,976 logits elements, `nonzero_count=151936`, sampled token `63690`,
  and diagnostic projection reference `max_abs_error=1.2e-07`.
- Passed corrected MPK-policy live A100 diagnostic:

  ```bash
  ARTIFACT=tmp/cuda-backend/qwen-full-rmsnorm-marker-fix-clean-mpk-2026-06-03
  PYTHONPATH=$PWD:$PWD/python timeout 480 .venv/bin/python \
    examples/cuda/qwen_decode_loop_runner.py --mode mock \
    --single-context-live-session --run-resource-backed-smoke \
    --resource-backed-task-selection first_layer_with_logits \
    --resource-backed-workload mpk_offline_decode \
    --resource-backed-repeat-runs 1 --resource-backed-decode-steps 1 \
    --resource-backed-worker-blocks 10 \
    --resource-backed-logits-check-policy final_step \
    --resource-backed-logits-active-cols full \
    --resource-backed-numeric-task-mode unit_math_full_rmsnorm \
    --device 0 --arch compute_80 \
    --cache-root $ARTIFACT/cache \
    --output-json $ARTIFACT/qwen-decode-loop-runner.json
  ```

- Corrected MPK result: `mpk_offline_decode` completed 10 resource-backed task
  functions with zero scheduler errors, `full_logits_buffer_checked` over
  2,430,976 logits elements, `nonzero_count=2430976`, sampled token `63690`,
  and diagnostic projection reference `max_abs_error=1.2e-07`.

## Remaining Gaps

This is still not full serving. PTO still needs efficient numerically correct
Qwen kernels, real full-serving decode execution, and imported full-serving
latency/throughput rows for MPK-policy and VDCores-policy workloads.
