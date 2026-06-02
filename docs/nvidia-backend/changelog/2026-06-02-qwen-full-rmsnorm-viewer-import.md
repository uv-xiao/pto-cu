# 2026-06-02 Qwen Full RMSNorm Viewer Import

## Code And Data Changed

- Added `full_reduction_contract_count` to the PTO Qwen resource-backed
  benchmark viewer importer.
- Extended the existing importer unit test with a compact
  `unit_math_full_rmsnorm` fixture so raw artifacts using
  `numeric_task_mode.full_reduction_contracts` keep explicit review evidence.
- Restored top-level benchmark validator exports used by focused review
  artifact tests after the validator split, and aligned the decode-loop runner
  review test with the current `full_cuda_live_decode_loop_execution` gap.

## Architecture Quality

The viewer import now records both RMSNorm variants: external-scale unit math
and full-reduction unit math. This keeps the benchmark status aligned with the
runtime mode split instead of hiding the full RMSNorm evidence inside raw JSON.

## Evaluation Run

- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_nvidia_changelog.py`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`.
- Passed:
  `PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/check_nvidia_review_ready.py`.

## Remaining Gaps

This is a viewer import evidence fix. It does not promote diagnostic
resource-backed Qwen rows to full-serving evidence; PTO still needs real
full-serving `mpk_offline_decode` and `vdcores_offline_decode` rows before the
paper-readiness gate can close.
