# 2026-06-01 Serving Coverage Guard

## Code And Data Changed

- Added required `statistic.serving_coverage` metadata to committed
  `llm_serving_decode` result rows.
- Updated the benchmark-viewer validator so every serving result row and every
  serving `viewer_result` evidence ref in the paper matrix declares the
  coverage class it represents.
- Updated serving result producers for MPK, SGLang, ThunderKittens, and the
  generic paper-baseline importer so future rows preserve or infer coverage.
- Added a `Coverage` column to the HTML result table.

## Architecture Quality

The review data now distinguishes full-serving evidence from proxy evidence in
a machine-checkable way. A shape match alone is no longer sufficient for
`llm_serving_decode` evidence. This protects the paper-readiness matrix from
accidentally treating a controlled attention-tile proxy, native bring-up, or
one-token diagnostic as a Qwen/Qwen3-8B full-serving row.

## Evaluation Run

No new performance benchmark was run. This was an evidence-contract update
over existing viewer results. The validation gate regenerates the audit, work
queue, and goal-progress JSON from the updated data.

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

Result: passed.

## Remaining Gaps

- PTO persistent-device still needs repo-owned Qwen/Qwen3-8B model loading,
  tokenization, KV-cache management, and decode-loop execution.
- VDCores and ThunderKittens-family full-serving rows remain queued in
  `paper_readiness_work_queue.json`.
