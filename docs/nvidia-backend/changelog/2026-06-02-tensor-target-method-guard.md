# Tensor Target Method Guard

## Code And Data Changed

- Threaded viewer method IDs into tensor workload coverage validation.
- Added a guard that rejects model-shape target methods that are not present
  in the benchmark viewer method catalog.

## Architecture Quality

The tensor model-shape plan now proves its comparison methods are renderable
viewer methods rather than unchecked strings. This keeps the target plan tied
to the human-reviewable benchmark surface.

## Evaluation Run

- Focused benchmark-viewer, changelog, review-readiness, and whitespace
  checks passed after adding the guard.

## Remaining Gaps

Backend implementation closure remains `in_progress` because tuned PTO tensor
bodies and normal PTO graph breadth still require implementation evidence.
