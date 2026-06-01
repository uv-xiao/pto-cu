# Paper Baseline Probe Shards

## Code And Data Changed

- Replaced the committed `paper_baseline_probes.json` payload with
  `data/paper_baseline_probes/index.json` and one record file per baseline
  source-entrypoint probe.
- Extended the shared viewer-data sharding helper for
  `paper_baseline_probes`.
- Updated the probe collector, probe-status updater, HTML viewer, and focused
  review tests to load the logical probe collection through sharded-aware
  helpers.

## Architecture Quality

Each baseline probe status is now reviewable as its own small JSON file. This
keeps A100/H200 machine-status evidence and blocking gaps local to the
baseline they describe.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

This is a reviewability cleanup only. It does not add new paired probe runs or
resolve existing baseline environment blockers.
