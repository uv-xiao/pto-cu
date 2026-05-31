# 2026-06-01 Serving Baseline Environment Plans

## Code And Data Changed

- Added `paper_baseline_environment_plan.py` to generate isolated vLLM and
  SGLang runtime-environment plans from the pinned paper-baseline source
  metadata.
- Added `paper_baseline_environment_plans.json` to the benchmark viewer data
  and the matching raw artifact under
  `tmp/cuda-backend/paper-baselines/environment-plans/`.
- Updated `paper_baseline_run_readiness.py` so vLLM and SGLang run-readiness
  records include an `environment_plan` check.
- Updated the benchmark viewer to render environment paths, dependency
  sources, critical packages, install commands, validation commands, and
  remaining execution gaps for each serving framework baseline.
- Updated the review guard and focused artifact tests to validate the new
  environment-plan contract.

## Architecture Quality

The serving framework setup is now represented as reviewable data instead of
being hidden inside ad hoc terminal commands. The plans keep large framework
dependencies out of the project `.venv`, require a dedicated `tmp/` venv, and
make `PYTHONNOUSERSITE=1` validation explicit so user-site packages cannot hide
missing runtime dependencies.

The vLLM plan also records `uvloop` as a manual package because the pinned
`api_server.py` imports it even though it is not declared in the inspected
runtime requirement files. That keeps the next install step honest instead of
assuming `pip install -r requirements/common.txt -r requirements/cuda.txt` will
clear every import blocker.

## Evaluation Run

Generated the environment-plan and derived review artifacts:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
```

The raw planner artifact is:

- `tmp/cuda-backend/paper-baselines/environment-plans/environment-plans-a4f5eabf/environment-plans.json`

The generated plans are `plan_ready` for:

- `vllm_runtime_environment`
- `sglang_runtime_environment`

Focused verification passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
  -q -k 'environment_plan or run_readiness_probe_exports_run_blockers or benchmark_viewer_has_json_backed_review_data'
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

`node --check docs/nvidia-backend/benchmark-viewer/viewer.js` also passed.

## Remaining Gaps

- The plans have not installed vLLM or SGLang dependencies yet. They are the
  reviewable recipe for the next execution slice.
- vLLM and SGLang run-readiness remain partial because the latest A100/H200
  probes still report missing installed framework modules and benchmark import
  dependencies.
- Actual serving benchmark runs still need raw JSON capture and import through
  `paper_baseline_results_update.py`.
