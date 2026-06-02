# 2026-06-02 PTO Submit Graph Builder

## Code And Data Changed

- Added `simpler_setup/cuda_pto_graph.py` with `CudaPtoTaskArg`,
  `CudaPtoTaskSubmit`, and `lower_cuda_pto_task_graph`.
- The new adapter mirrors PTO tag-driven dependency inference for CUDA
  persistent-device task graphs, then feeds the existing normal-graph lowering
  path.
- Updated the Qwen unit-math live CUDA example to build its persistent DAG from
  PTO-style tagged submits instead of hand-written normal-graph edges.

## Architecture Quality

The adapter keeps scene-test descriptor parsing separate from builder-side PTO
submit lowering. Reviewers can now inspect the CUDA dependency rule in a small
module instead of searching through `scene_test.py`.

## Evaluation Run

- Passed `py_compile` for the new adapter and updated Qwen graph module.
- Added and passed one focused unit test proving that tagged submits lower into
  persistent-device fan-in, dependent spans, and task records.
- Passed a no-CUDA Qwen graph-array import check confirming the converted
  example still emits `[0, 1, 1, 1]` fan-in and `[1, 2, 3]` dependents.

## Evidence

- `simpler_setup/cuda_pto_graph.py`
- `examples/cuda/qwen_unit_math_live_impl/graph.py`
- `tests/ut/py/test_cuda_backend.py`
- `docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`

## Remaining Gaps

This builder slice did not cover live C++ orchestrator snapshots or paired GPU
snapshot smoke; those were closed by later reports.
