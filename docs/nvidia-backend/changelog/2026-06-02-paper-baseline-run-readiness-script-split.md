# 2026-06-02 Paper Baseline Run Readiness Script Split

## Code And Data Changed

- Replaced `.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py`
  with a short CLI and compatibility export module.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness_impl/`.
- Preserved `build_run_readiness`, `load_json`, `write_json`, and default path
  constants from the original script path for refresh tooling.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps each run-readiness implementation module below the 300-line review
  target.
- Separates JSON I/O, baseline/probe/environment indexing, per-run readiness
  checks, and final payload assembly so reviewers can inspect each pre-run
  gate independently.
- Keeps the documented script path stable for planned MPK, VDCores, vLLM,
  SGLang, and ThunderKittens baseline execution preparation.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness_impl/*.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py \
    --output-root tmp/cuda-backend/paper-baselines/run-readiness/split-check-$(git rev-parse --short HEAD) \
    --viewer-output tmp/cuda-backend/paper-baselines/run-readiness/split-check-$(git rev-parse --short HEAD)/paper_baseline_run_readiness.json
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_run_readiness_probe_exports_run_blockers'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, CLI generation, focused pytest, diff check,
  changelog validation, and NVIDIA review guard passed.

## Remaining Gaps

- This split does not execute new paper-baseline runs or import new measured
  results. It improves the reviewability of the pre-run gate used before
  spending A100/H200 time on those runs.
