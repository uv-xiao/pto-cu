# 2026-05-31 Paper Evaluation Matrix

## Code And Data Changed

- Added
  `docs/nvidia-backend/benchmark-viewer/data/paper_evaluation_matrix.json`.
- Extended `.agents/checks/validate_benchmark_viewer_data.py` to validate the
  paper-evaluation matrix against benchmark, method, paper-baseline, and result
  identifiers.
- Updated the HTML benchmark viewer to load and render the paper-evaluation
  matrix as a separate review tab.
- Extended the NVIDIA review guard and focused artifact tests to require the
  matrix and its required claim coverage.
- Updated the shared contracts and evaluation plan to define how paper claims
  move from planned or partial evidence to paper-ready.

## Architecture Quality

The paper-readiness state is now structured data instead of prose scattered
across planning docs. Each claim names its workloads, methods, paper baselines,
hardware targets, required metrics, current evidence, missing evidence, and
promotion gate.

This makes it harder to accidentally describe a paper claim as ready when MPK,
VDCores, vLLM, SGLang, ThunderKittens, or required metric coverage is still
missing.

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

- The matrix deliberately marks several claims as partial or planned. Paper
  readiness still requires imported MPK, VDCores, vLLM, SGLang,
  ThunderKittens, direct CUDA, CUDA Graph, and tensor-core baseline result
  artifacts.
- Future evaluation importers should populate result records for those missing
  rows rather than editing the matrix by hand.
