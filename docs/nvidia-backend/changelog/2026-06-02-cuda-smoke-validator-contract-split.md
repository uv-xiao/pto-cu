# 2026-06-02 CUDA Smoke Validator Contract Split

## Summary

Moved CUDA smoke validator report-file and expectation contracts into a
focused helper module. The existing `cuda_validate_smoke.py` imports and
re-exports the same names, so tests, scripts, and CLI behavior stay stable.

## Code And Data Changed

- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke_impl/`
  as the helper package for smoke validation contracts.
- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke_impl/contracts.py`
  for `REPORT_FILES`, `ResourcePolicyExpectation`, and
  `SmokeValidationExpectation`.
- Updated
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py`
  to import those contracts from the helper package.

## Architecture Quality

- Separates public validation contracts from validation implementation logic.
- Keeps the helper module small and directly readable.
- Preserves the existing script-level names used by tests and command helpers.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py \
    .agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke_impl/contracts.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'cuda_smoke_validator_checks_resource_policy'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile check, focused resource-policy validator test, diff check,
  benchmark-viewer data validation, changelog validation, and NVIDIA review
  guard passed.

## Remaining Gaps

- This change does not split the remaining smoke-validation helper functions.
- It does not run a fresh CUDA smoke capture; contract behavior is covered by
  focused unit validation.
