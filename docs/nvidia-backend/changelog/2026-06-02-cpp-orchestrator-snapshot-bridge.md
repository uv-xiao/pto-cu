# 2026-06-02 C++ Orchestrator Snapshot Bridge

## What Changed

- Added `OrchestratorSubmitSnapshot` and `debug_next_level_submits` to the
  C++ hierarchical orchestrator.
- Exposed `_debug_next_level_submits` through the nanobind `_Orchestrator`
  binding.
- Added `cuda_pto_submit_from_orchestrator_snapshot` so CUDA persistent-device
  graph construction can consume live C++ NEXT_LEVEL task-slot snapshots.
- Added a focused test that submits live C++ orchestrator tasks, snapshots
  them, and lowers them through the CUDA PTO graph path.

## Architecture Quality

The bridge reads the existing task-slot state instead of duplicating
scene-test descriptor logic. It is valid before `drain()` resets the ring, so
reviewers can audit the exact C++ scheduling inputs that feed CUDA graph
construction.

## Evaluation

- Confirmed the new test failed before implementation because the snapshot
  conversion path was missing.
- Rebuilt the editable nanobind extension with
  `.venv/bin/pip install --no-build-isolation -e .`.
- Passed focused CUDA PTO graph tests, including live C++ orchestrator
  snapshot conversion.

## Evidence

- `src/common/hierarchical/orchestrator.h`
- `src/common/hierarchical/orchestrator.cpp`
- `python/bindings/worker_bind.h`
- `simpler_setup/cuda_pto_graph.py`
- `tests/ut/py/test_cuda_backend.py`
