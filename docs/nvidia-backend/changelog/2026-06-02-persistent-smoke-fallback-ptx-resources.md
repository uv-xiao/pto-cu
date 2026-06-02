# 2026-06-02 Persistent Smoke Fallback PTX Resources

## Summary

Moved the CUDA persistent smoke runner's embedded fallback PTX blobs into
separate resource files. The script still exposes the same fallback constants
and CLI behavior, but its Python logic is no longer mixed with generated PTX
data.

## Code And Data Changed

- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke_impl/`
  as the helper package for persistent smoke runner resources.
- Added
  `.agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke_impl/fallback_ptx.py`
  to load fallback PTX bytes from sibling resource files.
- Added resource PTX files for the direct vector-add, queue vector-add, and
  DAG `f32` fallback executors.
- Updated
  `.agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py`
  to import the fallback bytes from the helper package.

## Architecture Quality

- Keeps generated PTX data separate from persistent smoke runner control flow.
- Preserves the existing `embedded-sm80-*` source labels, so existing reports
  and viewer evidence remain stable.
- Reduces the main runner by about 480 lines without changing benchmark
  semantics.

## Evaluation Run

- Byte-for-byte PTX relocation check passed:

  ```bash
  .venv/bin/python - <<'PY'
  # Compared each resource against the matching embedded literal from HEAD.
  PY
  ```

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
    .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke_impl/fallback_ptx.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_cuda_benchmark_report.py \
    -q -k 'persistent_smoke_builds_graph_descriptor_dag_shape'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  ```

- Result: PTX body comparison, compile check, focused persistent smoke import
  test, benchmark-viewer data validation, diff check, changelog validation,
  and NVIDIA review guard passed.

## Remaining Gaps

- This change does not split the rest of `cuda_persistent_smoke.py`; it only
  removes generated fallback PTX blobs from the main runner.
- It does not run a fresh CUDA persistent-device smoke. Runtime semantics are
  covered by preserving the fallback PTX bodies, with only trailing blank EOF
  lines normalized for repository whitespace checks.
