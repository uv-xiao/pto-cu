# Tensor Model Shape Targets

## Code And Data Changed

- Added model-shape target tiles to tensor workload coverage viewer data.
- Rendered those targets in the Coverage tab with tile dimensions, model
  mapping, status, and runnable benchmark commands.
- Extended the benchmark-viewer validator so target tiles must satisfy WMMA
  shape constraints and list the required comparison methods.

## Architecture Quality

The tuned tensor workload gap is now reviewable as concrete shape work instead
of a vague tuning statement. The gap remains open until PTO tuned tensor bodies
produce multi-repeat rows for those shapes.

## Evaluation Run

- Focused benchmark-viewer, changelog, review-readiness, dispatch-log, and
  diff-whitespace checks passed after adding the target plan.

## Remaining Gaps

Backend implementation closure remains `in_progress` because tuned PTO tensor
bodies and normal PTO graph breadth still require implementation evidence.
