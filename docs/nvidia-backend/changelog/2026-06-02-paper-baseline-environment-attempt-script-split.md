# 2026-06-02 Paper Baseline Environment Attempt Script Split

## Code And Data Changed

- Replaced
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py`
  with a short CLI and compatibility export layer.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt_impl/`
  for attempt construction, JSON I/O, path helpers, environment-plan lookup,
  bounded command execution, and Git metadata.
- Preserved public helper imports such as `build_attempt`,
  `append_viewer_attempts`, `run_step`, `command_is_allowed`, and
  `plan_for_baseline`.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps every new environment-attempt implementation module below the 300-line
  review target.
- Separates environment-plan validation, bounded shell execution, viewer-output
  merge behavior, and artifact payload construction.
- Keeps CLI behavior and generated `paper_baseline_environment_attempts` JSON
  unchanged for benchmark-viewer consumers.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt_impl/*.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'environment_attempt_captures_bounded_steps or environment_attempt_appends_resume_window'
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_benchmark_viewer_data.py
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, focused environment-attempt pytest,
  benchmark-viewer data validation, diff check, changelog validation, and
  NVIDIA review guard passed.

## Remaining Gaps

- This split does not run new vLLM or SGLang environment setup attempts. It
  improves the runner structure used to record those attempts.
