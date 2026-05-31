# 2026-05-31 Imported Baseline Artifact Guard

## Code And Data Changed

- Tightened `validate_benchmark_viewer_data.py` so an
  `imported_to_viewer` paper-baseline run cannot reference missing
  `expected_artifacts`.
- Split ThunderKittens tensor-core evidence into imported bounded MHA capture
  (`thunderkittens_tile_kernel`) and planned full upstream sweep
  (`thunderkittens_full_sweep`).
- Regenerated paper-baseline run-readiness data and the paper-readiness audit.
- Added a focused regression test proving missing expected artifacts are
  rejected for imported paper-baseline runs.

## Architecture Quality

Imported paper-baseline run status now means the run's declared artifacts
exist locally under `tmp/`. Future raw outputs stay in separate planned run
records, so reviewers can distinguish current evidence from remaining work
without reading prose blockers.

## Evaluation Run

The guard was developed with a failing fixture first: the validator accepted an
`imported_to_viewer` run whose `expected_artifacts` included a missing JSON
file. After implementation, the fixture fails for the missing artifact and the
current viewer data validation passed once the ThunderKittens full sweep is
represented as a planned run.

## Remaining Gaps

This is an evidence guard and data split, not a new benchmark result. The
planned `thunderkittens_full_sweep` still needs H200 correctness and benchmark
captures before the tensor-core tile claim can be promoted.
