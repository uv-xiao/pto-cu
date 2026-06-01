# 2026-06-01 SGLang Fixed-Range H200 Repeats

## Code And Data Changed

- Added SGLang H200 fixed-range repeated-result rows for batches `1`, `2`,
  `4`, `8`, and `16` under the VDCores `128/64` serving policy.
- Extended the paper-baseline viewer importer so serving-repeat metrics keep
  standard deviation, min/max, request counts, token totals, and raw sample
  arrays instead of dropping them during normalization.
- Added a newer SGLang execution-attempt record that removes the prior repeat
  sample blocker from the online/offline path and keeps the remaining
  `bench_one_batch` blocker explicit.

## Architecture Quality

The repeated SGLang rows use the same fixed token-shape contract as the
previous batch ladder, but now report `sample_count=3` for each batch and mode.
The normalized viewer records include mean throughput plus raw sample arrays,
so outliers remain visible to reviewers.

The importer change is intentionally generic for paper-baseline serving data:
it preserves repeat statistics for future vLLM, MPK-equivalent, or PTO serving
rows without requiring hand-edited `results.json` records.

## Evaluation Run

The H200 checkout used the standalone pto-cu tree-sync fallback. No upstream
repository was edited or pushed.

Raw artifact:

- `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-fixedrange-repeats-eb75a235/`

Verification commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_nvidia_changelog.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/check_nvidia_review_ready.py
node --check docs/nvidia-backend/benchmark-viewer/viewer.js
```

Captured online serving output throughput:

| Batch | Samples | Mean tok/s | Stdev tok/s |
| ----: | ------: | ---------: | ----------: |
| 1 | 3 | 161.02159972046658 | 4.811789038056537 |
| 2 | 3 | 308.03817224015523 | 3.7900771249599527 |
| 4 | 3 | 597.9727093342293 | 18.009974529091934 |
| 8 | 3 | 1194.254514947911 | 14.368285107430397 |
| 16 | 3 | 2171.7566025420783 | 47.732620565818955 |

Captured offline-engine output throughput:

| Batch | Samples | Mean tok/s | Stdev tok/s |
| ----: | ------: | ---------: | ----------: |
| 1 | 3 | 156.19079316906286 | 2.707208317880333 |
| 2 | 3 | 288.1244764888034 | 1.504704224443859 |
| 4 | 3 | 554.4822850296894 | 11.523214519169507 |
| 8 | 3 | 1136.0633983460384 | 1.5730105337018707 |
| 16 | 3 | 1784.4900809591322 | 724.7257327028299 |

The offline batch-16 row intentionally preserves the raw throughput samples:
`2167.286434107662`, `2237.551058756805`, and `948.6327500129296` tok/s.

## Remaining Gaps

- Resolve the SGLang `bench_one_batch` `input_ids` `None` failure or document
  a paper policy that excludes the synthetic one-batch path.
- Align SGLang repeated rows with matching PTO persistent-device, MPK,
  VDCores, and vLLM serving-policy evidence before making paper claims.
