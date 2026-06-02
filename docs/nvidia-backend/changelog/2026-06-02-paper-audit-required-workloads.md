# Paper Audit Required Workloads

## Code And Data Changed

- Added `required_workload_ids` support for paper-readiness `viewer_result`
  evidence refs.
- Added a focused audit test proving one complete PTO full-serving row cannot
  satisfy a ref that requires both `mpk_offline_decode` and
  `vdcores_offline_decode`.
- Taught the benchmark-viewer matrix validator to reject malformed
  `required_workload_ids` values.

## Architecture Quality

The audit can now express multi-policy evidence requirements directly in a
viewer-result ref. This closes the gap where a single correct PTO
`Qwen/Qwen3-8B` policy row could satisfy a broad full-serving ref before both
paper comparison policies were imported.

## Evaluation Run

- Focused paper-readiness audit tests passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_paper_readiness_audit.py`.
- Python compile, benchmark-viewer data validation, changelog validation, and
  NVIDIA review guard passed for this slice.

## Remaining Gaps

The current matrix still keeps PTO full-serving evidence in
`missing_evidence_details` until real PTO rows exist for both required serving
policies.
