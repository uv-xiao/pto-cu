# 2026-05-31 Paper Baseline Runs

## Code And Data Changed

- Added
  `docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json`.
- Extended `.agents/checks/validate_benchmark_viewer_data.py` to validate
  paper-baseline reproduction runs against known paper baselines and paper
  evaluation matrix claims.
- Updated the HTML benchmark viewer to show reproduction runs under each paper
  baseline.
- Updated focused review artifact tests, the NVIDIA review guard, the
  evaluation plan, baseline survey, and shared contracts.

## Architecture Quality

Paper-baseline reproduction is now structured data rather than prose-only
instructions. Each MPK, VDCores, vLLM, SGLang, and ThunderKittens run declares
its setup commands, run commands, hardware target, workload policy, expected
tmp artifacts, required metrics, and viewer import target.

This gives future evaluation slices a stable contract for turning raw baseline
outputs into viewer result records.

## Evaluation Run

Expected verification for this report:

```bash
.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

node --check docs/nvidia-backend/benchmark-viewer/viewer.js

git diff --check
```

## Remaining Gaps

- These records are reproduction contracts, not completed baseline runs.
- Future slices must execute the commands on compatible hardware, capture raw
  outputs under `tmp/`, and add importers that convert those outputs into
  benchmark-viewer result rows.
