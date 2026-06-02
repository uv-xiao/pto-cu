# 2026-06-02 Normal Graph Scene Builder

## Code And Data Changed

- Added `persistent_dag_normal_graph_f32` to the CUDA persistent-device
  scene-test builder set.
- The new builder reads `normal_graph`, `normal_graph_path`, or
  `normal_graph_file` and materializes the existing persistent-device
  scheduler ABI.
- Added a focused unit test for a normal-graph fork/join input with dependency
  edges.
- Updated persistent scheduler coverage data and status docs to distinguish
  scene-test normal graph construction from broader PTO task-graph closure.

## Architecture Quality

The backend now has an explicitly named normal-graph construction surface
instead of relying only on `persistent_dag_graph_f32` descriptor terminology.
The runtime ABI is unchanged: normal graph config lowers into the same task
array, fan-in array, and dependent array used by the persistent scheduler.

## Evaluation Run

- Red check failed before implementation:
  `.venv/bin/python -m pytest -q tests/ut/py/test_cuda_scene_test.py::test_scene_test_builds_cuda_persistent_normal_graph_from_dependency_edges`
- Passed focused check:
  `.venv/bin/python -m pytest -q tests/ut/py/test_cuda_scene_test.py::test_scene_test_builds_cuda_persistent_normal_graph_from_dependency_edges`
- Passed compatibility check:
  `.venv/bin/python -m pytest -q tests/ut/py/test_cuda_scene_test.py::test_scene_test_builds_cuda_persistent_normal_graph_from_dependency_edges tests/ut/py/test_cuda_scene_test.py::test_scene_test_builds_cuda_persistent_graph_from_dep_gen_edges`
- Passed syntax check:
  `.venv/bin/python -m py_compile simpler_setup/scene_test.py`
- Passed JSON check:
  `.venv/bin/python -m json.tool docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`
- Passed guard checks:
  `.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`,
  `.venv/bin/python .agents/checks/check_nvidia_review_ready.py`, and
  `.venv/bin/python .agents/checks/validate_nvidia_changelog.py`.

## Remaining Gaps

- The persistent scheduler generalization page still requires paired A100/H200
  normal-graph evidence and normal PTO task-graph construction beyond
  scene-test graph config before it can be removed from `status.md`.
