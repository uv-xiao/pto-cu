# 2026-06-02 Triton Tensor Tile Capture Script Split

## Code And Data Changed

- Replaced
  `.agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture.py`
  with a short CLI and compatibility export layer.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture_impl/`
  for constants, JSON/path helpers, viewer-record conversion, Triton runtime
  capture, and latency statistics.
- Updated benchmark-viewer evidence references so `tensor_tile_kernel` and
  `tl.dot` point at the runtime capture module, while `viewer_record` points at
  the viewer-record conversion module.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps every new Triton tensor-tile capture module below the 300-line review
  target.
- Separates runtime capture from raw-artifact validation and viewer-row
  conversion.
- Keeps the CLI command path stable for benchmark-viewer run instructions and
  method evidence.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture.py \
    .agents/skills/cuda-backend-eval/scripts/triton_tensor_tile_capture_impl/*.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'triton_tensor_tile_capture_exports_fixture_records'
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, focused Triton fixture pytest, benchmark-viewer data
  validation, diff check, changelog validation, and NVIDIA review guard passed.

## Remaining Gaps

- This split does not run a new Triton CUDA capture. It improves the capture
  wrapper structure used to generate and convert those artifacts.
