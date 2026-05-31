# 2026-05-31 Paper Readiness Work Queue

## Code And Data Changed

- Added `paper_readiness_work_queue.py` to flatten audit `next_actions` into
  a generated viewer data file.
- Added committed `paper_readiness_work_queue.json` with 13 current work
  items across persistent-device, tensor-tile, and LLM-serving claims.
- Extended the benchmark viewer with a `Work Queue` tab that renders the
  priority, claim, source, owner, status, and action for each item.
- Tightened `validate_benchmark_viewer_data.py` so the committed queue must
  regenerate exactly from `paper_readiness_audit.json`.

## Architecture Quality

The paper-readiness path now has a single reviewable task table derived from
the audit instead of requiring reviewers to expand every claim and manually
collect next actions. The audit remains the source of truth; the work queue is
mechanically generated and validated against it.

## Evaluation Run

Focused TDD tests first failed because `paper_readiness_work_queue.py` and
`paper_readiness_work_queue.json` did not exist. After implementation, the
focused work-queue and viewer-data tests passed.

## Remaining Gaps

The queue is a planning and review artifact, not measured performance
evidence. The overall audit remains `not_paper_ready` until the listed raw
MPK, VDCores, vLLM, SGLang, ThunderKittens, and PTO serving captures are
executed and imported.
