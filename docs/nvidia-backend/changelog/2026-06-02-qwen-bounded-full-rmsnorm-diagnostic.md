# 2026-06-02 Qwen Bounded Full RMSNorm Diagnostic

## Code And Data Changed

- Capped `unit_math_full_rmsnorm` hidden-stage resource-backed packets to one
  4,096-element Qwen hidden vector.
- Exposed `element_limit=4096` in the full-RMSNorm numeric-mode contract.
- Imported the bounded A100 resource-backed diagnostic artifact into the
  benchmark viewer as two `diagnostic_resource_backed_qwen_dag` rows.

## Architecture Quality

The generated RMSNorm body still performs a per-output-element reduction, so
using the full MPK-policy batch buffer would make the diagnostic impractical.
The new cap keeps this path runnable while preserving the review distinction
between the external-scale RMSNorm shortcut and the generated reduction
branch. The rows remain diagnostic and cannot satisfy the full-serving gate.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_graph_materialization.py -k full_rmsnorm`.
- Passed live A100 diagnostic:
  `examples/cuda/qwen_decode_loop_runner.py` with
  `--resource-backed-numeric-task-mode unit_math_full_rmsnorm`,
  `--resource-backed-decode-steps 1`, and output artifact
  `tmp/cuda-backend/pto-serving-resource-backed-full-rmsnorm-2026-06-02/qwen-decode-loop-runner.json`.
- Result: both `mpk_offline_decode` and `vdcores_offline_decode` completed
  255 resource-backed tasks with zero scheduler errors and full logits-buffer
  diagnostic reference checks passing.

## Remaining Gaps

This is still not full serving. PTO still needs efficient numerically correct
Qwen kernels, real full-serving decode execution, and imported full-serving
latency/throughput rows for MPK-policy and VDCores-policy workloads.
