# 2026-06-01 Serving Baseline Environment Attempt

## Code And Data Changed

- Added `paper_baseline_environment_attempt.py` to execute bounded slices of a
  serving-baseline environment plan and capture per-step logs plus JSON
  evidence under `tmp/`.
- Added `paper_baseline_environment_attempts.json` to the benchmark viewer so
  humans can see which environment setup steps have actually run, not only the
  planned install recipe.
- Updated the benchmark viewer, viewer-data validator, and review artifact
  tests to require environment-attempt data, command logs, JSON artifacts, and
  the rule that environment commands do not install into `.venv` or use
  `pip --user`.

## Architecture Quality

The serving-baseline setup path now has a separable execution record. The
planner still owns the full vLLM/SGLang recipe, while the attempt script owns
bounded execution evidence with step status, command logs, artifact roots, and
remaining work. This keeps large framework installs out of the shared project
environment and makes partial progress reviewable without pretending the full
serving stack has been materialized.

## Evaluation Run

Ran the vLLM setup attempt for the first three environment-plan steps:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --max-steps 3 --timeout-seconds 300
```

The raw attempt artifact is:

- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-ef065acd/environment-attempt.json`

The captured attempt is `partial`: vLLM's dedicated `tmp/` venv was created,
`pip/setuptools/wheel` were upgraded inside it, and the explicit `uvloop`
package was installed. The remaining vLLM requirements, editable install, and
validation imports are still pending.

Focused verification passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
  -q -k 'environment_attempt_captures_bounded_steps or benchmark_viewer_has_json_backed_review_data'
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

- vLLM still needs the heavier `requirements/common.txt` and
  `requirements/cuda.txt` install steps, editable install, and import
  validation before serving benchmark execution.
- SGLang has an environment plan but no captured setup attempt yet.
- Actual serving benchmark runs still need raw JSON capture and import through
  `paper_baseline_results_update.py`.
