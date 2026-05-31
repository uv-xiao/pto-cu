# 2026-05-31 Paper Baseline Scheduler Metadata

## Code And Data Changed

- Extended `paper_baseline_viewer_export.py` so MPK and VDCores scheduler
  baseline rows can carry review-facing scheduler metadata into the benchmark
  viewer record.
- Added a focused fixture that imports both `mpk_persistent_scheduler_trace`
  and `vdcores_resource_policy_trace` rows and checks that
  `scheduler_overhead_ns`, `dispatch_trace`, `resource_policy`,
  `queue_pressure`, and `task_registry` survive normalization.
- Updated the CUDA evaluation skill to document the scheduler-specific metric
  fields accepted by the paper-baseline importer.

## Architecture Quality

The persistent-device comparison contract now has an explicit path for the
metadata that makes MPK and VDCores comparable to PTO CUDA scheduler captures.
The importer still accepts only named timing and scheduler fields instead of
copying arbitrary raw payload data into committed viewer records.

## Evaluation Run

The red/green fixture was verified with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py \
    -q -k 'paper_baseline_viewer_export_preserves_scheduler_trace_metadata'
```

The initial run failed with missing `scheduler_overhead_ns`; after the importer
change the focused fixture passed.

## Remaining Gaps

This change only preserves scheduler metadata once MPK or VDCores raw results
exist. It does not convert the planned MPK/VDCores scheduler runs into measured
viewer rows, and the paper-readiness audit should continue to show those
baselines as blockers until real raw artifacts are captured and imported.
