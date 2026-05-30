# 2026-05-31 Viewer Result Export

## Code And Data Changed

- Added `cuda_viewer_export.py` under the CUDA backend evaluation skill.
- Added `capture_imports.json` as the committed mapping from raw benchmark
  baselines to viewer benchmark and method IDs.
- Extended the viewer data validator to check capture-import mappings.
- Updated current viewer result records to use `median_capture_group` as the
  statistic kind emitted by the exporter.
- Added a focused artifact test that converts a fixture capture into
  viewer-schema result records.

## Architecture Quality

Raw CUDA benchmark JSON now has a repeatable path into the benchmark viewer.
The importer keeps raw artifacts under `tmp/`, reads committed mapping data,
and emits review-facing records with hardware, inputs, timing statistics,
sample count, correctness, and raw artifact path.

This reduces hand editing in future evaluation slices. When MPK, VDCores, or
new PTO CUDA benchmark families produce raw JSON, the expected extension point
is the mapping file plus importer logic rather than direct edits to
`results.json`.

## Evaluation Run

Expected verification for this report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py

.venv/bin/python -m py_compile \
  .agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py

git diff --check
```

The current compact capture can regenerate the committed `result_records`
with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py \
    tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/cuda-benchmark.json \
    --artifact-root tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/ \
    --output tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/viewer-result-records.json
```

## Remaining Gaps

- The exporter currently covers the committed PTO compact viewer rows. Future
  paper slices must extend the mapping for MPK, VDCores, vLLM, SGLang,
  ThunderKittens, and richer PTO CUDA sweeps.
- The committed viewer still stores selected microbenchmark rows. Full
  paper-grade statistics need imported raw artifacts with repeated samples,
  throughput, scheduler-overhead breakdowns, and baseline-specific metadata.
