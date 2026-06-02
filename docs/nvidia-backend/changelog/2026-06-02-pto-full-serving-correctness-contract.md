# PTO Full-Serving Correctness Contract

## Code And Data Changed

- Tightened
  `.agents/skills/cuda-backend-eval/scripts/pto_qwen_full_serving_viewer_import.py`
  so PTO full-serving imports require `correctness_details`, not only
  top-level `correctness=pass`.
- Tightened
  `.agents/skills/cuda-backend-eval/scripts/paper_readiness_audit_impl/claim_status.py`
  so hand-written viewer rows must carry the same full-Qwen correctness
  evidence before closing the PTO paper-readiness item.
- Updated the CUDA evaluation workflow notes and paper work-queue promotion
  text to require explicit full-Qwen numerical correctness evidence.
- Added focused importer and paper-audit tests for the new evidence contract.

## Architecture Quality

The benchmark viewer now separates a generic pass flag from paper-ready Qwen
correctness. PTO rows for `llm_serving_decode` / `pto_persistent_device` must
name `full_qwen_numerical_correctness`, `Qwen/Qwen3-8B`, a passing status,
token match, positive checked-token count, and finite error within tolerance.
The importer normalizes these fields into both `correctness_details` and
`statistic`, and the audit requires the two copies to agree.

## Evaluation Run

- Focused importer and audit tests passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest
  tests/ut/py/test_nvidia_pto_full_serving_viewer_import.py
  tests/ut/py/test_nvidia_paper_readiness_audit.py
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py -q`.

## Remaining Gaps

The contract is enforced, but the PTO paper-readiness item still needs real
MPK-policy and VDCores-policy full-serving raw rows whose full-Qwen correctness
details pass this importer.
