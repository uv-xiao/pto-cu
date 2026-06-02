# Backend Gap Status Sync Guard

## Code And Data Changed

- Updated the goal-progress validator to parse `docs/nvidia-backend/status.md`
  and require exact agreement with the generated backend-closure evidence refs.
- Updated the focused review artifact test to check the same status-to-viewer
  synchronization contract.

## Architecture Quality

The backend closure criterion can no longer drift from the status landing page.
When a remaining gap is added, closed, or reclassified, the generated viewer
data and tests must change with the status archive.

## Evaluation Run

- Focused pytest passed:
  `1 passed, 61 deselected`.
- `validate_benchmark_viewer_data.py` passed.
- `validate_nvidia_changelog.py` passed.
- `check_nvidia_review_ready.py` passed.

## Remaining Gaps

Backend implementation closure remains `in_progress`; this change only makes
the remaining-gap evidence exact and self-checking.
