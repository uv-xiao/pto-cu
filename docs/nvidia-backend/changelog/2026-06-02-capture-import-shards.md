# Capture Import Shards

## Code And Data Changed

- Replaced the committed `capture_imports.json` payload with
  `data/capture_imports/index.json` and one record file per raw-capture import
  rule.
- Extended the shared viewer-data sharding helper for `capture_imports`.
- Updated the CUDA viewer export script and focused review tests to load the
  logical capture-import collection through sharded-aware helpers.

## Architecture Quality

Each raw benchmark capture import rule is now reviewable as its own small JSON
record. This keeps the mapping from raw capture baselines to viewer benchmark
and method IDs local to each capture shape.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

This is a reviewability cleanup only. It does not add new raw capture imports
or change existing benchmark evidence.
