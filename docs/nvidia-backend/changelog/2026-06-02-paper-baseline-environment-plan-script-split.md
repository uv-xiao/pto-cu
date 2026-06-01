# 2026-06-02 Paper Baseline Environment Plan Script Split

## Code And Data Changed

- Replaced
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan.py`
  with a short CLI and compatibility export layer.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan_impl/`
  for artifact building, dependency parsing, JSON I/O, paths, environment
  specs, and Git metadata.
- Preserved the public imports consumed by
  `refresh_nvidia_review_artifacts.py`, including `DEFAULT_BASELINES`,
  `DEFAULT_OUTPUT_ROOT`, `DEFAULT_VIEWER_OUTPUT`, `load_json`, `write_json`,
  `build_environment_plan`, and `build_environment_plans`.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps every new environment-plan implementation module below the 300-line
  review target.
- Separates serving-framework environment policy from TOML/requirements
  dependency evidence and plan assembly.
- Keeps the CLI behavior and generated JSON schema unchanged for the benchmark
  viewer and paper-baseline run-readiness tooling.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan.py \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan_impl/*.py
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_environment_plan_exports_isolated_serving_envs'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, focused environment-plan pytest, diff check,
  changelog validation, and NVIDIA review guard passed.

## Remaining Gaps

- This split does not materialize vLLM or SGLang environments. It only makes
  the existing environment-plan generator easier to review before future
  remote evaluation work.
