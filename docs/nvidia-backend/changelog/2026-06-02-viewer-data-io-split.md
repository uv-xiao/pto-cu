# 2026-06-02 Viewer Data I/O Split

## Code And Data Changed

- Replaced `.agents/skills/cuda-backend-eval/scripts/viewer_data_io.py` with
  a short compatibility shim.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/viewer_data_io_impl/` for
  constants, record naming, sidecars, reads, and writes.
- Preserved the original `load_json` and `write_json` imports used by
  benchmark-viewer exporters and paper-baseline refresh tooling.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps every viewer-data I/O implementation module below the 300-line review
  target.
- Separates sharded collection naming, sidecar handling, load logic, and write
  logic so reviewers can inspect the benchmark-viewer data contract in smaller
  units.
- Keeps the logical `.json` path contract stable for sharded collections such
  as results, paper-baseline probes, run readiness, readiness audit, and capture
  imports.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/viewer_data_io.py \
    .agents/skills/cuda-backend-eval/scripts/viewer_data_io_impl/*.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'benchmark_viewer_has_json_backed_review_data'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, benchmark-viewer data validation, focused pytest,
  diff check, changelog validation, and NVIDIA review guard passed.

## Remaining Gaps

- This split does not add new benchmark results or paper-baseline captures. It
  improves the reviewability of the shared data path that writes and reads
  those artifacts.
