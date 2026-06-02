# PTO Full-Serving Row Gate

## Code And Data Changed

- Added a row-level PTO Qwen full-serving promotion check to
  `.agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py`.
- The gate now reports missing MPK and VDCores policy rows, plus per-row
  missing requirements for diagnostic Qwen rows already present in the viewer.
- Added focused tests for rejecting diagnostic resource-backed Qwen rows and
  accepting complete MPK/VDCores policy rows.

## Architecture Quality

The preflight now makes the full-serving evidence contract explicit in code:
PTO rows must be `llm_serving_decode` / `pto_persistent_device`, name
`Qwen/Qwen3-8B`, use `full_serving` coverage, pass correctness, carry a serving
policy ID, and include paper latency/throughput metrics. Diagnostic rows remain
reviewable progress but cannot close the paper-readiness item.

## Evaluation Run

- Targeted preflight tests:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_review_artifacts.py -k 'pto_serving_preflight or pto_full_serving_row_gate'`.
- Python compile, benchmark-viewer data validation, changelog validation, and
  NVIDIA review guard passed for this slice.

## Remaining Gaps

The PTO paper-readiness item still needs real full-serving Qwen/Qwen3-8B rows
for both `mpk_offline_decode` and `vdcores_offline_decode`.
