# 2026-05-31 Ultimate Goal Setup

## Code And Data Changed

- Added the standalone pto-cu NVIDIA backend ultimate-goal contract under
  `docs/in_progress/`.
- Added dispatcher preparation, shared benchmark/evidence contracts, and a
  paper-ready evaluation plan.
- Added a baseline source survey and paper-baseline viewer data for MPK,
  VDCores, vLLM, SGLang, and ThunderKittens.
- Promoted vLLM, SGLang, and ThunderKittens from planned source capture to
  cloned-for-survey status with pinned commits and first reproduction commands.
- Extended the NVIDIA review artifact test to require the ultimate-goal docs.
- The review guard now checks the new goal artifacts.

## Architecture Quality

The new docs make ownership boundaries explicit: stable NVIDIA backend docs
describe accepted behavior, `docs/in_progress/` owns dispatcher state, viewer
JSON owns benchmark status, and changelog reports own human-readable deltas.

The goal also records the key CUDA architecture constraint: the
persistent-device runtime cannot rely on an AICPU launcher, so its scheduler
must be compiled into the CUDA device binary with the task implementations.

The baseline survey separates source readiness from result readiness. MPK,
VDCores, vLLM, SGLang, and ThunderKittens are now recorded as
cloned-for-survey systems, while measured performance remains a future child
slice that must produce raw artifacts before appearing in result tables.

## Evaluation Run

Before adding the artifacts, the focused review test failed because the
ultimate-goal files and changelog did not exist. The expected verification is:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

git diff --check
```

## Remaining Gaps

- The current branch sets the paper-ready contract; it does not complete the
  full MPK, VDCores, or LLM-serving baseline evaluation.
- Future child PRs must expand viewer schemas, capture raw benchmark data, and
  reproduce or compare against MPK, VDCores, and their paper baselines.
