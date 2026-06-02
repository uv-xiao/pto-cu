# 2026-06-02 CUDA Benchmark Fallback PTX Resources

## Summary

Moved the CUDA benchmark script's generated slow vector-add fallback PTX into
a resource file. The benchmark still uses the same `_compile_slow_ptx()` path
and keeps the `embedded-sm80-slow-ptx` report label when `nvcc` is missing.

## Code And Data Changed

- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark_impl/`
  as the helper package for CUDA benchmark resources.
- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark_impl/fallback_ptx.py`
  to load fallback PTX bytes from sibling resource files.
- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark_impl/`
  `fallback_ptx/slow_vector_add_sm80.ptx`
  for the slow vector-add fallback executor.
- Updated
  `.agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py`
  to import the fallback bytes from the helper package.

## Architecture Quality

- Keeps generated PTX data separate from benchmark orchestration and report
  rendering logic.
- Reduces the already-large `cuda_benchmark.py` script without changing the
  fallback compile contract.
- Keeps the new helper and PTX resource files small enough for review.

## Evaluation Run

- PTX body comparison passed:

  ```bash
  .venv/bin/python - <<'PY'
  # Compared the resource body against the matching embedded literal from HEAD.
  PY
  ```

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    .agents/skills/cuda-backend-eval/scripts/cuda_benchmark_impl/fallback_ptx.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'summarize_results_groups_by_machine_and_baseline'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: PTX body comparison, compile check, focused benchmark import test,
  diff check, benchmark-viewer data validation, changelog validation, and
  NVIDIA review guard passed.

## Remaining Gaps

- This change does not split the rest of `cuda_benchmark.py`; it only removes
  generated fallback PTX data from the main script.
- It does not run a fresh CUDA benchmark capture. Runtime semantics are covered
  by preserving the fallback PTX body and by import-level validation.
