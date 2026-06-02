# 2026-06-02 PTO full-serving metric gate

## Code And Data Changed

- Added full-serving metric/accounting validation to `.agents/skills/cuda-backend-eval/scripts/pto_qwen_full_serving_viewer_import.py`.
- Added focused importer tests for failed-request rejection and underchecked decode-token rejection in `tests/ut/py/test_nvidia_pto_full_serving_viewer_import.py`.
- Added review artifact `tmp/cuda-backend/pto-full-serving-metric-gate-2026-06-02/full-serving-metric-gate.json`.
- Added current evidence shard `docs/nvidia-backend/benchmark-viewer/data/paper_evaluation_matrix/records/llm_serving_paper_baselines/current_evidence_refs/items/079.json`.

## Architecture Quality

The PTO full-serving viewer import path now rejects raw rows unless the row proves a complete serving run: at least three samples, zero failed requests, completed requests equal to batch size, optional input/output token totals match the declared prompt/decode lengths, and correctness checks cover every generated decode token. This keeps diagnostic or partial captures out of paper-facing viewer results.

## Evaluation Run

- `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_pto_full_serving_viewer_import.py -q` passed.

## Remaining Gaps

This change strengthens the evidence gate only. PTO persistent-device Qwen3-8B still needs real full-serving MPK and VDCores rows before `pto_full_serving_qwen3_8b` can become paper-ready.
