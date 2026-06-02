# PTO Full-Serving Viewer Import

## Code And Data Changed

- Added `pto_qwen_full_serving_viewer_import.py` for importing real PTO
  `Qwen/Qwen3-8B` full-serving rows into benchmark-viewer results.
- The importer requires both `mpk_offline_decode` and
  `vdcores_offline_decode`, correctness pass, a `tmp/` raw artifact root, and
  the latency/throughput metrics required by the paper-readiness gate.
- Added focused tests for valid row generation, missing-policy rejection, and
  sharded viewer-result merging.

## Architecture Quality

The importer follows the existing viewer-data sharding path instead of adding a
parallel storage convention. It emits rows with
`method_id=pto_persistent_device`, `serving_coverage=full_serving`, and stable
serving workload IDs, so the audit and matrix gates can verify PTO evidence
without special-casing raw artifacts.

## Evaluation Run

- Focused importer tests passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_pto_full_serving_viewer_import.py`.
- Python compile, benchmark-viewer data validation, changelog validation, and
  NVIDIA review guard passed for this slice.

## Remaining Gaps

The importer is ready, but the actual PTO full-serving raw rows still require
numerically correct Qwen kernel execution for both paper serving policies.
