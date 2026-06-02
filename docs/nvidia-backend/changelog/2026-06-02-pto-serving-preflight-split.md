# PTO Serving Preflight Split

## Code And Data Changed

- Split `pto_serving_preflight.py` into a thin CLI wrapper plus focused helper
  modules under `pto_serving_preflight_impl/`.
- Kept the exported row-gate helpers used by existing tests:
  `row_workload_id`, `full_serving_qwen_row_status`, and
  `full_serving_qwen_rows`.
- Updated the decode-loop preflight fixture so its positive row satisfies the
  strict full-serving policy instead of only setting
  `serving_coverage=full_serving`.

## Architecture Quality

The PTO full-serving preflight gate is now easier to review: JSON/scaffold IO,
viewer-row validation, checklist construction, and payload assembly live in
separate files. Each file stays below the repository's soft size target while
preserving the same CLI and evidence contract.

## Evaluation Run

- Python compile passed for the preflight wrapper and helper modules.
- Direct preflight capture passed:
  `.venv/bin/python .agents/skills/cuda-backend-eval/scripts/pto_serving_preflight.py --output tmp/cuda-backend/pto-serving-preflight/refactor-check.json`.
- Focused tests passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_review_artifacts.py -k 'pto_serving_preflight or pto_full_serving_row_gate'`.
- Decode-loop preflight fixture test passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_decode_loop_runner.py -k 'preflight_does_not_promote'`.

## Remaining Gaps

This is a guardrail quality change. PTO persistent-device still needs real
`Qwen/Qwen3-8B` full-serving raw rows for `mpk_offline_decode` and
`vdcores_offline_decode` before the paper-readiness queue can promote that
claim.
