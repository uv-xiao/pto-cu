# 2026-06-02 TaskArgs PTO Submit Bridge

## Code And Data Changed

- Extended `simpler_setup/cuda_pto_graph.py` with
  `cuda_pto_submit_from_task_args`.
- The bridge reads real PTO `TaskArgs` tensors and `TensorArgType` tags, then
  creates CUDA persistent-device submit records for normal graph lowering.
- Added a focused unit test using real `ContinuousTensor`, `TaskArgs`, and
  `TensorArgType` bindings.

## Architecture Quality

This moves the CUDA persistent-device builder boundary closer to the live PTO
orchestrator path without expanding `scene_test.py`. The remaining gap is now
specifically live C++ orchestrator task-slot capture, not Python
`TaskArgs`/tag conversion.

## Evaluation Run

- Passed `py_compile` for `simpler_setup/cuda_pto_graph.py`.
- Passed focused CUDA backend tests for PTO submit graph lowering and real
  `TaskArgs` conversion.

## Evidence

- `simpler_setup/cuda_pto_graph.py`
- `tests/ut/py/test_cuda_backend.py`
- `docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`

## Remaining Gaps

This bridge covered real `TaskArgs` conversion but still left live C++
orchestrator snapshot capture to later reports.
