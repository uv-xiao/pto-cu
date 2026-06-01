# 2026-06-02 Benchmark Viewer JavaScript Split

## Code And Data Changed

- Replaced the long benchmark viewer script with a small `viewer.js`
  entrypoint and focused ES modules under `benchmark-viewer/viewer/`.
- Updated `benchmark-viewer/index.html` to load the viewer as a browser module.
- Fixed the viewer loader so `paperBaselineEnvironmentPlans` is loaded into
  state before the baseline-readiness view renders it.
- Updated the NVIDIA review guard and review-artifact test to scan the viewer
  module tree instead of only the entrypoint.

## Architecture Quality

- Keeps each viewer implementation file below the 300-line review target.
- Separates data loading, DOM helpers, lookup helpers, summary rendering,
  catalog rendering, baseline rendering, paper-readiness rendering, and tab
  wiring.
- Preserves the existing JSON data contract and rendered section IDs.

## Evaluation Run

- Focused validation passed:

  ```bash
  for f in docs/nvidia-backend/benchmark-viewer/viewer.js \
    docs/nvidia-backend/benchmark-viewer/viewer/*.js; do \
    node --check "$f" || exit 1; \
  done
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/checks/check_nvidia_review_ready.py \
    .agents/checks/nvidia_review_guard/*.py \
    tests/ut/py/test_nvidia_review_artifacts.py
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'benchmark_viewer_has_json_backed_review_data or \
    ultimate_goal_artifacts_define_paper_ready_cuda_path'
  ```

- Result: viewer data validation and review guard passed; focused pytest
  passed with `2 passed, 60 deselected`.
- Browser smoke was not run because Playwright and Chromium are not available
  in this local environment.

## Remaining Gaps

- This split does not add new benchmark data. It improves the maintainability
  of the human-reviewable benchmark viewer and fixes one existing loader
  wiring gap.
