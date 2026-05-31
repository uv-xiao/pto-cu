# 2026-05-31 Paper Baseline Run Readiness

## Code And Data Changed

- Added `paper_baseline_run_readiness.py` under the CUDA backend evaluation
  scripts.
- Added `paper_baseline_run_readiness.json` for the benchmark viewer.
- Rendered run-readiness records next to paper-baseline reproduction runs.
- Extended the benchmark-viewer validator, review guard, focused tests, and
  evaluation docs to require MPK/VDCores scheduler-run readiness coverage.

## Architecture Quality

The remaining MPK/VDCores scheduler blockers now have a reviewable pre-run
state instead of only a planned command. The readiness data records source
entrypoints, expected artifact paths, required metrics, model-access state, and
VDCores extension-build state. It is explicitly separate from benchmark
results, so it cannot accidentally promote unmeasured baselines.

## Evaluation Run

The run-readiness probe was developed with a failing fixture first:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_run_readiness_probe_exports_run_blockers'
```

The real current readiness artifact was generated with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_run_readiness.py \
    --output-root tmp/cuda-backend/paper-baselines/run-readiness/run-readiness-3157ea68 \
    --viewer-output docs/nvidia-backend/benchmark-viewer/data/paper_baseline_run_readiness.json
```

Current MPK readiness is `partial` because `HF_TOKEN` is not available in the
local environment. Current VDCores readiness is `partial` because `HF_TOKEN`
is not available and `python/dae/runtime*.so` has not been built in the
source checkout.

## Remaining Gaps

Readiness records do not replace measured MPK/VDCores scheduler results. The
planned scheduler runs must still be executed on appropriate hardware, captured
under `tmp/cuda-backend/paper-baselines/`, imported through
`paper_baseline_results_update.py`, and then reflected in the paper-readiness
audit.
