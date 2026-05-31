# 2026-05-31 Paper Readiness Next Actions

## Code And Data Changed

- Extended `paper_readiness_audit.py` so each blocked paper-readiness claim
  carries generated `next_actions`.
- Added viewer-data validation for audit next-action records and their
  source-specific identifiers.
- Updated the benchmark viewer to render a `Next Actions` list beside each
  paper-readiness audit claim.
- Regenerated `paper_readiness_audit.json` from the current matrix,
  run-readiness, probe, and result data.

## Architecture Quality

The audit now separates why a claim is blocked from what the next operator step
is. Matrix gaps, run-readiness records, and readiness probes remain the source
of truth, while the generated audit becomes a review-facing work queue instead
of a prose-only blocker summary.

## Evaluation Run

The focused TDD test first failed because the committed audit and viewer did
not expose `next_actions`. After implementation, focused tests and the
viewer-data validator passed. The full verification set for this slice is
recorded in the PR update.

## Remaining Gaps

The audit still reports the overall paper-readiness status as
`not_paper_ready`. The generated next actions point to the remaining MPK,
VDCores, vLLM, SGLang, ThunderKittens, and PTO persistent-device raw captures
that must be imported before paper promotion.
