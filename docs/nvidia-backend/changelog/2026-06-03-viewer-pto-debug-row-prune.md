# 2026-06-03 Viewer PTO Debug Row Prune

## Code And Data Changed

- Removed three committed PTO `llm_serving_decode` result shards that only
  represented older proxy, descriptor-smoke, or unit-math debug evidence.
- Kept the current resource-backed PTO Qwen rows in the committed viewer data
  because they are stronger evidence for the implementation path under review.
- Updated `record_files.json` and the result index count from 44 to 41.

## Architecture Quality

The benchmark viewer now keeps review-facing result records focused on
representative measured rows and current resource-backed Qwen diagnostics. Raw
and historical debug artifacts remain under `tmp/` and in prior changelog
reports instead of growing the committed viewer data.

## Evaluation Run

No new benchmark was run for this cleanup. The change is a committed-data
compaction only. Verification:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py

PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_nvidia_changelog.py
```

## Remaining Gaps

The PTO Qwen rows are still diagnostic, not full-serving correctness rows.
Full paper-ready import still requires full Qwen token/logit correctness for
policy-length MPK and VDCores serving workloads.
