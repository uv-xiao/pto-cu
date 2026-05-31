# 2026-05-31 Review Artifact Refresh

## Code And Data Changed

- Added `refresh_nvidia_review_artifacts.py` as the single command for
  regenerating generated NVIDIA review JSON.
- Added a regression test that regenerates the paper-readiness audit, work
  queue, and goal-progress audit into a temporary directory and compares them
  with committed viewer data.
- Updated the CUDA evaluation skill and shared contracts to make the unified
  refresh command the normal workflow.

## Architecture Quality

Generated review artifacts now have an explicit dependency-order refresh path:
audit first, work queue second, goal progress last. This reduces the chance of
reviewers updating one generated file while leaving dependent files stale.

## Evaluation Run

Focused TDD first failed because `refresh_nvidia_review_artifacts.py` did not
exist. After implementation, the focused refresh regression passed.

## Remaining Gaps

This is review-infrastructure work. It does not reduce the 13 queued
paper-readiness actions or import new raw baseline captures.
