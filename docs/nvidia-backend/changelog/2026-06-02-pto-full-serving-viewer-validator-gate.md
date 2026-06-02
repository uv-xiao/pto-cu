# 2026-06-02 PTO full-serving viewer validator gate

## Code And Data Changed

- Added PTO Qwen full-serving result validation under `.agents/checks/benchmark_viewer_validation/pto_full_serving.py`, called from `.agents/checks/benchmark_viewer_validation/results.py`.
- Added focused validator tests in `tests/ut/py/test_nvidia_benchmark_viewer_result_validation.py`.
- Added review artifact `tmp/cuda-backend/pto-full-serving-viewer-validator-gate-2026-06-02/viewer-validator-gate.json`.
- Added current evidence shard `docs/nvidia-backend/benchmark-viewer/data/paper_evaluation_matrix/records/llm_serving_paper_baselines/current_evidence_refs/items/081.json`.

## Architecture Quality

The human-reviewable benchmark viewer now rejects weak PTO `full_serving` Qwen rows before they enter published viewer data. This complements the raw importer and paper-readiness audit gates by validating the viewer JSON boundary itself.

## Evaluation Run

- `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_benchmark_viewer_result_validation.py -q` passed.
- `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py` passed.

## Remaining Gaps

This change only strengthens viewer-data quality control. PTO persistent-device full-serving MPK/VDCores rows for `Qwen/Qwen3-8B` are still missing.
