# 2026-06-01 vLLM Environment Materialization

## Code And Data Changed

- Made `paper_baseline_environment_attempt.py` resumable with
  `--start-step`, `--attempt-id-suffix`, and `--append-viewer` so long
  environment setup can be captured as multiple reviewable execution windows.
- Updated `paper_baseline_environment_plan.py` so vLLM installs declared CUDA
  build requirements from `requirements/build/cuda.txt` before editable
  installation.
- Tightened environment setup commands to run pip with
  `PYTHONNOUSERSITE=1` and the isolated environment's `bin/` first in `PATH`.
- Updated benchmark-viewer data to show vLLM setup attempts for steps 1-6,
  including the current editable-install failure.
- Strengthened viewer-data validation and focused tests for resumed attempt
  windows, user-site isolation, and env-local build-tool resolution.

## Architecture Quality

The serving-baseline environment flow now separates plan generation from
bounded execution attempts. That makes long setup work restartable and lets
reviewers see exactly which plan step passed or failed without losing earlier
evidence. The install recipe also no longer accepts user-site packages or user
`PATH` build tools as hidden dependencies, which keeps paper-baseline setup
closer to a reproducible environment contract.

The failed vLLM editable install is now a source/build compatibility blocker,
not an environment-planning blocker: runtime requirements and CUDA build
requirements install successfully in the isolated env before the build reaches
`spinloop.cpp`.

## Evaluation Run

Regenerated review artifacts, recreated the vLLM environment under
`tmp/cuda-backend/paper-baselines/envs/vllm-27fa5aa3`, and ran bounded setup
windows:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --max-steps 3 --timeout-seconds 300 --commit 1d3242de
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --start-step 4 --max-steps 1 \
  --attempt-id-suffix step04 --append-viewer \
  --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step04 \
  --timeout-seconds 900 --commit 1d3242de
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --start-step 5 --max-steps 1 \
  --attempt-id-suffix step05 --append-viewer \
  --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step05 \
  --timeout-seconds 600 --commit 1d3242de
```

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/paper_baseline_environment_attempt.py \
  --baseline vllm --start-step 6 --max-steps 1 \
  --attempt-id-suffix step06 --append-viewer \
  --output-root tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step06 \
  --timeout-seconds 900 --commit 1d3242de
```

Current raw artifacts:

- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de/environment-attempt.json`
- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step04/environment-attempt.json`
- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step05/environment-attempt.json`
- `tmp/cuda-backend/paper-baselines/environment-attempts/vllm-1d3242de-step06/environment-attempt.json`

Results:

- Steps 1-3 passed: venv creation, build-tool upgrade, and explicit `uvloop`.
- Step 4 passed: vLLM runtime/CUDA requirements installed with user-site
  disabled.
- Step 5 passed: vLLM CUDA build requirements installed with env-local build
  tools first in `PATH`.
- Step 6 failed after entering the actual vLLM editable build. The current
  blocker is `csrc/spinloop.cpp` failing under `Py_LIMITED_API=0x030b0000`
  because `Py_buffer` and `PyBuffer_Release` are not visible.

## Remaining Gaps

- vLLM is not installed yet; serving runs remain blocked on the step-6 build
  failure.
- The next vLLM slice should decide whether to patch the pinned source locally
  for reproducibility, change build flags to avoid the limited API for
  `spinloop`, or use an upstream-supported prebuilt/skip-extension mode if one
  exists and is valid for the paper baseline.
- SGLang still has an environment plan but no captured setup attempt.
- Actual vLLM and SGLang serving benchmark runs still need raw JSON capture and
  import through `paper_baseline_results_update.py`.
