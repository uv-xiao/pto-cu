# 2026-06-02 Qwen Block RMSNorm Diagnostic

## Code And Data Changed

- Changed generated `qwen_rmsnorm_input` from element-threaded to block-threaded.
- Replaced the per-output RMSNorm reduction with one shared-memory block
  reduction per task, followed by a block-stride normalized writeback.
- Removed the temporary `unit_math_full_rmsnorm` hidden-element cap so
  resource-backed diagnostics use the full workspace hidden extent.

## Architecture Quality

The persistent-device scheduler still launches the same task function ids, but
the RMSNorm callable now owns the block-wide reduction work inside the generated
device binary. This keeps the `unit_math_full_rmsnorm` mode on the CUDA
persistent-device architecture path without adding a host-side special case.

## Evaluation Run

- Passed focused packet/source tests:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_graph_materialization.py -k full_rmsnorm`
  and
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_task_body_math.py -k generated_source_contains_qwen_unit_math_kernels`.
- Passed live A100 diagnostic:
  `examples/cuda/qwen_decode_loop_runner.py` with
  `--resource-backed-numeric-task-mode unit_math_full_rmsnorm`,
  `--resource-backed-decode-steps 1`, and output artifact
  `tmp/cuda-backend/pto-serving-resource-backed-full-rmsnorm-block-2026-06-02/qwen-decode-loop-runner.json`.
- Result: both `mpk_offline_decode` and `vdcores_offline_decode` completed
  255 resource-backed tasks with zero scheduler errors.

## Remaining Gaps

This is still diagnostic evidence. PTO still needs numerically complete Qwen
kernels and full-serving latency/throughput rows for MPK-policy and
VDCores-policy workloads.
