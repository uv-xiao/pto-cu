# 2026-06-02 Normal Graph Coverage Status

## Code And Data Changed

- Added `normal_graph_lowering_boundary` to
  `docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`.
- Made the benchmark-viewer validator require that coverage group.
- Updated persistent scheduler status docs so the remaining gap is
  backend-builder normal PTO graph construction plus paired A100/H200 evidence,
  not the shared edge-lowering primitive.

## Architecture Quality

The status now distinguishes three layers: scheduler mechanics, shared
normal-graph edge lowering, and backend-builder graph construction. This keeps
reviewers from reading the remaining scheduler gap as a missing fan-in or
dependent-span lowering primitive.

## Evaluation Run

- Passed syntax check:
  `.venv/bin/python -m py_compile .agents/checks/benchmark_viewer_validation/persistent_scheduler_coverage.py`
- Passed JSON check:
  `.venv/bin/python -m json.tool docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`
- Passed guard check:
  `.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py`
- Passed guard check:
  `.venv/bin/python .agents/checks/check_nvidia_review_ready.py`
- Passed guard check:
  `.venv/bin/python .agents/checks/validate_nvidia_changelog.py`

## Remaining Gaps

- Full backend-builder normal PTO graph construction and paired A100/H200
  normal-graph evidence still remain open under the persistent scheduler
  generalization status page.
