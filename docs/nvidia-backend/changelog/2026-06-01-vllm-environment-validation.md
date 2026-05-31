# 2026-06-01 vLLM Environment Validation

## Code And Data Changed

- Updated the vLLM editable-install command to derive `VLLM_VERSION_OVERRIDE`
  from the pinned `tmp/baselines/vllm` checkout before building the copied
  source overlay.
- Updated `vllm_spinloop_source_overlay.py` to omit copied `.deps/` build
  cache directories so CMake FetchContent state is regenerated for the overlay
  path.
- Added an env-local `scipy>=1.15.0` install before vLLM validation imports,
  because the project rule uses `--system-site-packages` and the host SciPy is
  incompatible with vLLM's NumPy dependency.
- Refreshed the benchmark viewer environment-plan and environment-attempt data
  with the successful editable install and validation import windows.

## Architecture Quality

The vLLM serving baseline now has a reviewable local materialization path:
runtime/build requirements install in a dedicated `tmp/` venv, source
modifications live only in a copied overlay, package version metadata still
comes from the pinned upstream checkout, and validation imports run with
user-site disabled.

This is still an environment-readiness result, not a serving benchmark result.
The paired source probes and final serving commands remain separate evidence
layers.

## Evaluation Run

Ran the clean overlay install window:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --start-step 6 --max-steps 3 \
  --attempt-id-suffix step06_overlay_preflight_install_clean --append-viewer \
  --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-e1c48975-step06-overlay-preflight-install-clean \
  --timeout-seconds 1800 --commit e1c48975
```

Results:

- step 6 overlay recreation passed in `1.425s`;
- step 7 overlay preflight passed in `0.098s`;
- step 8 editable install passed in `760.226s`;
- installed vLLM version: `0.1.dev1+g27fa5aa3b`.

Ran the validation window:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --start-step 9 --max-steps 5 \
  --attempt-id-suffix step09_scipy_validation --append-viewer \
  --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-e1c48975-step09-scipy-validation \
  --timeout-seconds 300 --commit e1c48975
```

Results:

- step 9 installed env-local `scipy-1.15.3`;
- `vllm` import passed;
- `vllm.entrypoints.cli.main` import passed;
- `vllm.entrypoints.openai.api_server` import passed;
- `vllm.engine.arg_utils` import passed.

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-e1c48975-step06-overlay-preflight-install-clean/environment-attempt.json`
- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-e1c48975-step09-scipy-validation/environment-attempt.json`

## Remaining Gaps

- vLLM serving and throughput benchmark commands still need raw JSON capture
  and viewer import.
- The current successful vLLM materialization is local A100 evidence. H200
  serving-baseline environment validation remains separate work.
- SGLang environment materialization remains pending.
