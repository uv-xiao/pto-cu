# 2026-05-31 PTO Tensor-Core Viewer Evidence

## Code And Data Changed

- Added a `capture_imports.json` rule for
  `pto_persistent_dag_graph_tensor_core`.
- Imported A100 and H200 `tensor_core_tile` viewer rows for
  `pto_persistent_device` from the compact paired capture at
  `tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/`.
- Updated the tensor-core paper-evaluation matrix so the PTO row is current
  evidence instead of a missing-evidence blocker.
- Regenerated `paper_readiness_audit.json`.

## Architecture Quality

The imported row stays on the same reviewable abstraction as the rest of the
viewer: one explicit capture-import rule maps a named raw benchmark baseline to
one benchmark, method, shape, dtype, and repeat policy. The raw PTO tensor-core
artifact remains under `tmp/`; committed data only records normalized viewer
rows and the matrix evidence reference.

## Evaluation Run

The focused TDD check first failed because the exporter did not create a
`tensor_core_tile` row for `pto_persistent_device`. After adding the import rule
and viewer rows, the readiness audit reports the tensor-core claim with three
remaining blockers instead of four, and no missing viewer result for the PTO
row. The focused exporter/review-artifact pytest target passed after the data
refresh.

## Remaining Gaps

- CUTLASS or CuTe has not been captured for the same tile shape.
- ThunderKittens still needs full upstream correctness and benchmark sweeps
  beyond the bounded MHA capture.
