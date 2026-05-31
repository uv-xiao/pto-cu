# 2026-06-01 vLLM Spinloop Preflight

## Code And Data Changed

- Added `vllm_spinloop_preflight.py` to detect the pinned vLLM spinloop
  stable-ABI incompatibility before starting the long editable build.
- Added `preflight_commands` and `preflight_after_install_steps` to the
  serving-baseline environment plan contract.
- Updated `paper_baseline_environment_attempt.py` so environment attempts run
  install steps, preflight steps, remaining install steps, then validation
  steps in a reviewable order.
- Updated the benchmark viewer and validator so preflight commands and
  preflight attempt steps are visible and checked.
- Refreshed vLLM environment-attempt data so step 6 now records the preflight
  failure instead of rerunning the long editable-install failure.

## Architecture Quality

The vLLM setup contract now separates environment materialization from source
compatibility gating. Runtime dependencies and build dependencies can still be
installed in the isolated `tmp/` environment, but the known Python 3.10
limited-API mismatch is caught as a fast deterministic preflight before the
editable build.

This keeps the upstream vLLM checkout untouched. The next fix can be reviewed
as either an environment decision, such as using Python 3.11+, or a standalone
local reproducibility patch/build flag if that is chosen for paper-baseline
execution.

## Evaluation Run

Regenerated review artifacts and replayed vLLM environment setup windows at
commit `94ba2e06`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --max-steps 3 --timeout-seconds 300 --commit 94ba2e06
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --start-step 6 --max-steps 1 \
  --attempt-id-suffix step06 --append-viewer \
  --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-94ba2e06-step06 \
  --timeout-seconds 60 --commit 94ba2e06
```

Current raw artifact:

- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-94ba2e06-step06/environment-attempt.json`

The step-6 preflight failed in `0.108s` with the explicit blocker:

- `spinloop uses Py_buffer/PyBuffer_Release while the target is built with USE_SABI 3.11, but the isolated environment uses Python 3.10.12 headers.`

Focused verification passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
  -q -k 'vllm_spinloop_preflight or environment_plan_exports_isolated_serving_envs or environment_attempt_appends_resume_window'
```

## Remaining Gaps

- vLLM still is not installed. The next vLLM slice must resolve the preflight
  blocker by using Python 3.11+ for the baseline environment or by adding a
  reviewed local reproducibility patch/build flag that removes `Py_LIMITED_API`
  from the spinloop CXX compile.
- Validation imports and serving benchmark commands remain blocked until the
  editable install succeeds.
- SGLang environment materialization remains pending.
