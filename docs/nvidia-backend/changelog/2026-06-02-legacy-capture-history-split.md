# 2026-06-02 Legacy Capture History Split

## Code And Data Changed

- Replaced `docs/nvidia-backend/history/captures/legacy-captures.md` with a
  short archive landing page.
- Added focused child pages under
  `docs/nvidia-backend/history/captures/legacy-captures/` for source
  artifacts, launch/vector sweeps, DAG shape notes, reproduction commands, and
  next evaluation gaps.
- Added dispatch-log evidence for the split in
  `docs/in_progress/nvidia_backend_paper_ready/dispatch_log/entries/`.

## Architecture Quality

- Keeps the stable legacy capture path required by the review guard while
  moving the large archived body into small, intent-scoped pages.
- Reduces the largest NVIDIA evaluation-history document from a long mixed
  archive into a reviewable map plus focused subdocuments.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/checks/check_nvidia_review_ready.py \
    .agents/checks/nvidia_review_guard/*.py \
    tests/ut/py/test_nvidia_review_artifacts.py
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  PYTEST_K='ultimate_goal_artifacts_define_paper_ready_cuda_path or '
  PYTEST_K+='benchmark_viewer_has_json_backed_review_data or '
  PYTEST_K+='evaluation_docs_are_split_for_review'
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k "$PYTEST_K"
  ```

- Result: review guard passed; focused pytest passed with `3 passed,
  59 deselected`.

## Remaining Gaps

- This split does not add new benchmark data. It only improves the
  human-reviewable organization of existing historical CUDA captures.
