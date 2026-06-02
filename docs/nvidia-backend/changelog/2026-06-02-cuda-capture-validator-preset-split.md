# 2026-06-02 CUDA Capture Validator Preset Split

## Summary

Moved the paper-ready CUDA capture validator preset data out of
`cuda_validate_capture.py` and into a focused preset module. The validator CLI,
`validate_capture()`, and the `paired-current` / `compact-current` preset
contracts remain unchanged.

## Code And Data Changed

- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture_impl/`
  as the helper package for capture validation data.
- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture_impl/presets.py`
  for preset machines, baselines, sizes, report files, graph metadata, and
  expected paper-source IDs.
- Updated
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py`
  to import preset data from the helper package.

## Architecture Quality

- Separates large paper-ready preset tables from validation logic.
- Keeps the new preset module below the 300-line review target.
- Makes future preset changes easier to review without scanning the full
  capture validator implementation.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture.py \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_capture_impl/presets.py
  SELECTOR='cuda_capture_validator_paired_current_requires_generic_args_baseline or cuda_capture_validator_compact_current_preset_matches_docs_gate'
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k "$SELECTOR"
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile check, focused paired-current and compact-current preset
  tests, diff check, benchmark-viewer data validation, changelog validation,
  and NVIDIA review guard passed.

## Remaining Gaps

- This change does not split the remaining capture-validation helper
  functions; it only moves preset data out of the validator.
- It does not run a fresh CUDA benchmark capture. Preset behavior is covered
  by focused unit tests.
