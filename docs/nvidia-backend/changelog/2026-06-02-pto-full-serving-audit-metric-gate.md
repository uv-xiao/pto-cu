# 2026-06-02 PTO full-serving audit metric gate

## Code And Data Changed

- Added audit-side PTO full-serving request/token accounting checks to `.agents/skills/cuda-backend-eval/scripts/paper_readiness_audit_impl/claim_status.py`.
- Added focused audit tests for nonzero failed requests and insufficient checked decode tokens in `tests/ut/py/test_nvidia_paper_readiness_audit.py`.
- Added review artifact `tmp/cuda-backend/pto-full-serving-audit-metric-gate-2026-06-02/audit-metric-gate.json`.
- Added current evidence shard `docs/nvidia-backend/benchmark-viewer/data/paper_evaluation_matrix/records/llm_serving_paper_baselines/current_evidence_refs/items/080.json`.

## Architecture Quality

The paper-readiness audit no longer trusts a PTO viewer row solely because it is labeled `full_serving`. It independently requires complete request and token accounting before a PTO `Qwen/Qwen3-8B` row can satisfy the full-serving evidence reference.

## Evaluation Run

- `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_paper_readiness_audit.py -q` passed.

## Remaining Gaps

This change prevents overclaiming in generated review data. It does not add the missing real PTO persistent-device full-serving MPK/VDCores rows.
