# 2026-06-02 Python Orchestration Graph Recorder

## Code And Data Changed

- Added `CudaPtoGraphRecorder` and `record_cuda_pto_submits` to
  `simpler_setup/cuda_pto_graph.py`.
- The recorder runs a Python orchestration function and captures
  `submit_next_level` calls as CUDA persistent-device submit records.
- Added a focused unit test that records two real `TaskArgs` submissions from
  an orchestration function and lowers them into fan-in/dependent arrays.

## Architecture Quality

The recorder gives the CUDA persistent-device builder a path from actual
Python orchestration code without expanding `scene_test.py`. The remaining
promotion gate is now live C++ hierarchical orchestrator task-slot capture and
paired A100/H200 evidence for that live path.

## Evaluation Run

- Confirmed the new recorder test failed before implementation because
  `record_cuda_pto_submits` did not exist.
- Passed `py_compile` for `simpler_setup/cuda_pto_graph.py`.
- Passed focused PTO graph tests covering tagged submits, real `TaskArgs`
  conversion, and Python orchestration recording.

## Evidence

- `simpler_setup/cuda_pto_graph.py`
- `tests/ut/py/test_cuda_backend.py`
- `docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`

## Remaining Gaps

This recorder slice still left live C++ task-slot snapshot capture and paired
GPU snapshot evidence to later reports.
