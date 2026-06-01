# 2026-06-02 Baseline Survey Split

## Code And Data Changed

- Replaced `docs/in_progress/nvidia_backend_paper_ready/baseline_survey.md`
  with a short stable landing page.
- Added focused baseline-survey pages under
  `docs/in_progress/nvidia_backend_paper_ready/baseline_survey/` for source
  state, MPK/VDCores, vLLM/SGLang, ThunderKittens, and PTO comparison actions.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps the stable `baseline_survey.md` path used by review guards, goal
  progress data, and paper-matrix evidence.
- Keeps every baseline-survey page below the 300-line review target.
- Separates baseline source readiness from PTO comparison mapping so MPK,
  VDCores, serving-framework, and ThunderKittens updates can be reviewed
  independently.

## Evaluation Run

- Focused validation passed:

  ```bash
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'ultimate_goal_artifacts_define_paper_ready_cuda_path'
  ```

- Result: changelog validation and NVIDIA review guard passed; focused pytest
  passed with `1 passed, 61 deselected`.

## Remaining Gaps

- This split does not add new baseline runs. It improves the reviewability of
  existing paper-baseline source readiness and comparison mapping.
