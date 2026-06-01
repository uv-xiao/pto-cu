# Paper Baseline Run Shards

## Code And Data Changed

- Replaced the committed `paper_baseline_runs.json` payload with
  `data/paper_baseline_runs/index.json` and one record file per paper-baseline
  run.
- Extended the shared viewer-data sharding helper for `paper_baseline_runs`.
- Updated the viewer, result importer, run-readiness probe, viewer exporter,
  serving-command planner, and focused review tests to load the logical runs
  collection through sharded-aware helpers.

## Architecture Quality

Each paper-baseline run contract is now inspectable as a small JSON record.
This makes setup commands, required metrics, import targets, and measurement
scope reviewable without scrolling through one large benchmark-viewer file.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

This is a reviewability cleanup only. It does not add new MPK, VDCores,
ThunderKittens, vLLM, SGLang, or PTO serving captures.
