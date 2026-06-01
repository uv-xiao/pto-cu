# Execution Attempt Shards

## Code And Data Changed

- Replaced the monolithic
  `docs/nvidia-backend/benchmark-viewer/data/paper_baseline_execution_attempts.json`
  with `data/paper_baseline_execution_attempts/index.json` plus one record file
  per execution attempt under `data/paper_baseline_execution_attempts/records/`.
- Added sharded-collection loading to the HTML viewer and the Python readiness
  readers used by the audit refresh and paper-baseline import flow.
- Updated review tests to load the execution-attempt collection through the
  shard index.

## Architecture Quality

The execution-attempt ledger is still one logical collection, but future
baseline changes now touch a single short record file. The largest current
record is kept below the repo's review-size target by listing key review
artifacts while preserving the complete raw run tree through `artifact_root`.

## Evaluation Run

- Regenerated `paper_readiness_audit.json`,
  `paper_readiness_work_queue.json`, and `goal_progress.json`.
- Validation is covered by `validate_benchmark_viewer_data.py`, which now
  verifies the shard index and every referenced record file.

Command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

This is a reviewability improvement only. The paper-readiness work queue still
blocks on full PTO Qwen serving, VDCores Qwen3-8B correctness, and
ThunderKittens-family full-serving rows.
