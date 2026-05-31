# 2026-06-01 vLLM Spinloop Overlay

## Code And Data Changed

- Added `vllm_spinloop_source_overlay.py` to create a copied vLLM source tree
  under `tmp/cuda-backend/paper-baselines/source-overlays/`.
- Updated the vLLM environment plan so editable install, validation imports,
  and the spinloop preflight use the copied build source instead of mutating
  `tmp/baselines/vllm`.
- Added viewer and validator coverage for `build_source_path` and
  `source_overlay_commands`.
- Refreshed the vLLM environment plan and added a bounded step-6/7 attempt
  showing overlay creation and overlay preflight both pass.

## Architecture Quality

The vLLM Python 3.10 reproducibility path is now explicit and reviewable. The
upstream checkout remains a pinned source input, while the build source is a
separate generated copy under `tmp/` with a single recorded build-flag change:
`-UPy_LIMITED_API` for the spinloop CXX compile. This keeps paper-baseline
execution moving without silently editing upstream source state.

The benchmark viewer now shows both the source overlay command and the build
source path, so a reviewer can distinguish provenance from the local
reproducibility copy used for installation.

## Evaluation Run

Regenerated the environment plan at commit `460a34ba`:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_plan.py \
  --commit 460a34ba \
  --output-root tmp/cuda-backend/paper-baselines/environment-plans/environment-plans-460a34ba
```

Ran the bounded overlay and preflight window:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --start-step 6 --max-steps 2 \
  --attempt-id-suffix step06_overlay_preflight --append-viewer \
  --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-460a34ba-step06-overlay-preflight \
  --timeout-seconds 300 --commit 460a34ba
```

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/source-overlays/vllm-27fa5aa3-spinloop-cpython/pto-cu-source-overlay.json`
- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-460a34ba-step06-overlay-preflight/environment-attempt.json`

Results:

- step 6 overlay creation passed in `4.489s`;
- step 7 spinloop preflight on the overlay passed in `0.091s`;
- the bounded attempt stops before editable install, so vLLM remains partial.

Focused verification passed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
  -q -k 'environment_plan_exports_isolated_serving_envs or vllm_spinloop'
```

## Remaining Gaps

- vLLM editable install from the overlay still needs to be run and captured.
- vLLM validation imports and serving benchmark commands remain pending until
  editable install succeeds.
- SGLang environment materialization remains pending.
