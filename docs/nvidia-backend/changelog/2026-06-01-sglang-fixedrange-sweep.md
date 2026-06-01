# 2026-06-01 SGLang Fixed-Range H200 Sweep

## Code And Data Changed

- Added SGLang H200 fixed-range batch `2`, `4`, `8`, and `16` online-serving
  rows to the benchmark viewer for the VDCores `128/64` policy.
- Added matching SGLang offline-engine throughput rows for the same batch
  ladder.
- Added a newer SGLang execution-attempt record that replaces the previous
  missing-batch-ladder blocker with the remaining `bench_one_batch` and repeat
  sample blockers.

## Architecture Quality

The SGLang batch ladder now uses the same fixed token-shape contract as the
batch-1 import: `random-ids`, `--tokenize-prompt`, and
`--random-range-ratio 1.0` for online serving, plus offline throughput with
context length `384` and `--skip-warmup`.

The viewer still keeps this as a partial paper-baseline attempt because the
online/offline sweep is a single sample and does not repair the
`bench_one_batch` synthetic-input failure. That separation avoids presenting a
complete SGLang baseline while preserving the useful batch-ladder evidence.

## Evaluation Run

The H200 checkout used the standalone pto-cu tree-sync fallback. No upstream
repository was edited or pushed.

Raw artifact:

- `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-fixedrange-sweep-cfbdcf0c/`

Verification commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_nvidia_changelog.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/check_nvidia_review_ready.py
node --check docs/nvidia-backend/benchmark-viewer/viewer.js
```

Captured online serving metrics:

| Batch | Completed | Failed | Output tok/s | Mean TTFT ms | Mean ITL ms |
| ----: | --------: | -----: | -----------: | -----------: | ----------: |
| 2 | 2 | 0 | 173.9169706505443 | 313.52585554122925 | 6.519593421369791 |
| 4 | 4 | 0 | 484.47103616685513 | 128.4353609662503 | 6.041537764374597 |
| 8 | 8 | 0 | 1177.5193163104066 | 45.52369890734553 | 5.874532896092172 |
| 16 | 16 | 0 | 1793.5576766141346 | 115.7339365745429 | 6.806777699463529 |

Captured offline-engine metrics:

| Batch | Successful | Output tok/s | Total latency s |
| ----: | ---------: | -----------: | --------------: |
| 2 | 2 | 288.98312805396284 | 0.4429324329830706 |
| 4 | 4 | 562.3668373531619 | 0.4552188767120242 |
| 8 | 8 | 1141.1108121634202 | 0.4486856092698872 |
| 16 | 16 | 2170.7243728274034 | 0.47173193097114563 |

## Remaining Gaps

- Resolve the SGLang `bench_one_batch` `input_ids` `None` failure or document
  a paper policy that excludes the synthetic one-batch path.
- Add repeated SGLang samples for variance and confidence intervals.
- Align final SGLang rows with matching PTO persistent-device, MPK, VDCores,
  and vLLM serving-policy evidence before making paper claims.
