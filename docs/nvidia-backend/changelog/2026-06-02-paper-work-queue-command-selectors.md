# Paper Work Queue Command Selectors

## Code And Data Changed

- Added `serving_command_plan_selectors` to paper-readiness work-queue items.
- Updated the work-queue builder to intersect audit serving targets with each
  baseline run's actual serving scope before emitting command-plan selectors.
- Tightened benchmark-viewer validation so every emitted selector resolves to
  at least one sharded serving command-plan record.
- Added a `Command Plan` column to the paper work-queue viewer table.

## Architecture Quality

Queued paper-baseline serving work now has a reviewable path from missing
evidence, to baseline run, to generated command-plan records. The guard also
prevents stale audit scope from implying runnable commands that do not exist;
ThunderKittens now points only at its VDCores-serving command plan.

## Evaluation Run

Focused verification passed:

- `validate_benchmark_viewer_data.py`
- `validate_nvidia_changelog.py`
- `check_nvidia_review_ready.py`
- `pytest -q test_paper_readiness_work_queue_matches_current_audit`
- `git diff --check`

## Remaining Gaps

The selectors expose the commands needed to continue paper-baseline work. They
do not replace the remaining VDCores, ThunderKittens, or PTO full-serving
result imports listed in the work queue.
