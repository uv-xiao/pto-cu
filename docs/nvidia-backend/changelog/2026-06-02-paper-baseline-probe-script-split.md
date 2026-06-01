# 2026-06-02 Paper Baseline Probe Script Split

## Code And Data Changed

- Replaced `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py`
  with a short CLI and compatibility export module.
- Added focused implementation modules under
  `.agents/skills/cuda-backend-eval/scripts/paper_baseline_probe_impl/`.
- Preserved the original script path used by paired A100/H200 probe tooling.
- Added dispatch-log evidence for the split.

## Architecture Quality

- Keeps each paper-baseline probe implementation module below the 300-line
  review target.
- Separates default paths, JSON I/O, shell-command helpers, per-check execution,
  and raw-probe artifact assembly so reviewers can inspect the source/dependency
  readiness gate independently.
- Keeps the CLI stable for local A100 and remote H200 probe capture before
  expensive paper-baseline runs.

## Evaluation Run

- Focused validation passed:

  ```bash
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m py_compile \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_probe_impl/*.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/skills/cuda-backend-eval/scripts/paper_baseline_probe.py \
    --output tmp/cuda-backend/paper-baselines/probes/split-check-$(git rev-parse --short HEAD)/probe.json \
    --artifact-root tmp/cuda-backend/paper-baselines/probes/split-check-$(git rev-parse --short HEAD)/
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_probe_collects_source_readiness'
  git diff --check
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/validate_nvidia_changelog.py
  PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
    .agents/checks/check_nvidia_review_ready.py
  ```

- Result: compile checks, CLI probe generation, focused pytest, diff check,
  changelog validation, and NVIDIA review guard passed.

## Remaining Gaps

- This split does not update committed probe statuses or run new paired A100/H200
  probes. It improves the reviewability of the probe generator used before
  those captures.
