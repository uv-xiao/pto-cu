# Result Record Shards

## Code And Data Changed

- Replaced the committed 7,813-line benchmark-viewer `results.json` payload
  with `data/results/index.json`, `data/results/record_files.json`, and one
  short file per `result_records` row under `data/results/records/`.
- Added `viewer_data_io.py` so result importers can read and write the sharded
  collection without recreating the monolithic file.
- Updated the HTML viewer, readiness audit, review guards, and focused tests to
  treat `results.json` as a logical path backed by the sharded collection.

## Architecture Quality

The benchmark viewer still exposes one logical `result_records` table, but
future evidence imports now touch only the affected result row files. This
makes paper-review diffs smaller while preserving raw artifact links and the
existing import target contract.

## Evaluation Run

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

This is a reviewability and guardrail improvement only. Paper readiness still
requires full PTO Qwen serving rows, VDCores Qwen3-8B correctness, and
ThunderKittens-family full-serving comparison rows.
