# Dispatcher Command Selector Guard

## Code And Data Changed

- Tightened dispatcher-backlog validation so every
  `serving_command_plan_selectors` value in the generated paper work queue
  must appear in the dispatcher backlog.
- Updated the active paper work-item list to show the VDCores and
  ThunderKittens command-plan selectors that reviewers can match against the
  Serving Commands viewer tab.

## Architecture Quality

The resumable dispatcher backlog now carries the same runnable command-plan
keys as the generated work queue, so a later session can move from paper
work item to command plan without re-deriving the baseline/run/workload tuple.

## Evaluation Run

Focused verification passed:

- `validate_benchmark_viewer_data.py`
- `validate_nvidia_changelog.py`
- `check_nvidia_review_ready.py`
- `git diff --check`

## Remaining Gaps

This makes queued work easier to resume. It does not produce the remaining
PTO, VDCores, or ThunderKittens full-serving result imports.
