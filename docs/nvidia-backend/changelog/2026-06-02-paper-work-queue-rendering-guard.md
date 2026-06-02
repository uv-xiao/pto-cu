# Paper Work Queue Rendering Guard

## Code And Data Changed

- Updated the benchmark viewer paper work queue table to render work item id,
  paper baseline, paper-baseline run, execution-attempt id, and item-level
  promotion gate.
- Tightened review and unit-test guards so those fields stay visible in the
  JSON-backed viewer.

## Architecture Quality

Reviewers can now trace each active paper blocker from the visible work queue
to the owning baseline/run or diagnostic attempt, and can see the promotion
gate without opening raw JSON.

## Evaluation Run

- `check_nvidia_review_ready.py` passed with the new rendering contract.
- Focused viewer review-data pytest passed after the table update.

## Remaining Gaps

This improves reviewability of active paper blockers. The work queue still has
four active items and remains `not_paper_ready`.
