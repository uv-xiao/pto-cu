# 2026-06-03 Benchmark Viewer Result Compaction

## Code And Data Changed

- Compacted committed benchmark-viewer result shards from 144 records to 44
  current representative records.
- Kept the rows needed by the paper-evaluation matrix, serving-workload
  evidence references, current PTO Qwen diagnostics, and baseline comparison
  rows.
- Left raw and historical captures under `tmp/`; the compaction decision record
  is under `tmp/cuda-backend/viewer-data-compaction-2026-06-03/`.

## Architecture Quality

The benchmark viewer remains the human-review surface, but committed JSON now
stores the current review set instead of repeated historical runs. This reduces
PR size without weakening the evidence contract: required viewer rows still
load through the same sharded `results.json` path and must pass validation.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: `benchmark viewer data validation passed`.

## Remaining Gaps

This only trims committed viewer data. It does not close the remaining full
Qwen serving correctness and paper-baseline evaluation gaps.
