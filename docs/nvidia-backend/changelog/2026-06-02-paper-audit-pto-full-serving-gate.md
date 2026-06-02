# Paper Audit PTO Full-Serving Gate

## Code And Data Changed

- Tightened paper-readiness audit evidence matching for PTO persistent-device
  Qwen full-serving viewer rows.
- Added focused audit tests for rejecting weak PTO rows that only have
  `full_serving` coverage and accepting complete PTO policy rows with
  correctness, workload ID, raw artifact, latency, and throughput fields.

## Architecture Quality

The audit no longer depends on shape and coverage alone for PTO
`Qwen/Qwen3-8B` full-serving evidence. It now mirrors the stricter preflight
contract before a row can unblock paper-readiness work, while leaving non-PTO
baseline matching unchanged.

## Evaluation Run

- Focused paper-readiness audit tests passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_paper_readiness_audit.py`.
- Python compile, benchmark-viewer data validation, changelog validation, and
  NVIDIA review guard passed for this slice.

## Remaining Gaps

The audit gate still waits for real PTO full-serving Qwen/Qwen3-8B rows for
both `mpk_offline_decode` and `vdcores_offline_decode`.
