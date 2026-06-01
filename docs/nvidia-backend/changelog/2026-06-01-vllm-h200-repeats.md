# 2026-06-01 vLLM H200 Repeats

## Code And Data Changed

- Added vLLM H200 repeated-result rows for the VDCores `128/64` serving
  policy and the MPK `64/1024` serving policy, both on `Qwen/Qwen3-8B`.
- Imported ten viewer rows: batches `1`, `2`, `4`, `8`, and `16` for each
  serving policy, with `sample_count=3` per row.
- Updated the vLLM run contract to point at the imported repeat artifacts
  instead of the older placeholder `tmp/cuda-backend/paper-baselines/vllm/`
  paths.
- Replaced the stale vLLM probe blocker for H200 module imports with a current
  H200 pass artifact. The probe remains partial because A100 runtime
  validation was not rerun for this H200 paper-baseline capture.

## Architecture Quality

The vLLM rows reuse the same paper-baseline serving result schema as the
SGLang repeat rows. Each row preserves mean throughput, standard deviation,
min/max, token totals, request counts, and raw throughput samples, so reviewer
analysis can see variance instead of only a single batch-ladder point.

The import remains evidence-driven: raw benchmark outputs stay under `tmp/`,
the checked-in viewer data stores normalized rows, and the readiness queue now
removes the older vLLM repeated-sample execution blocker.

## Evaluation Run

The H200 checkout used the standalone pto-cu tree-sync fallback. No upstream
repository was edited or pushed.

Raw artifact:

- `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-qwen3-8b-repeats-eb75a235/`

Verification commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
PYTHONPATH=$PWD:$PWD/python .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python .agents/checks/validate_nvidia_changelog.py
PYTHONPATH=$PWD:$PWD/python .agents/checks/check_nvidia_review_ready.py
node --check docs/nvidia-backend/benchmark-viewer/viewer.js
```

Captured VDCores-shaped vLLM output throughput:

| Batch | Samples | Mean tok/s | Stdev tok/s |
| ----: | ------: | ---------: | ----------: |
| 1 | 3 | 155.0161459156807 | 13.91467734153101 |
| 2 | 3 | 312.86978656376226 | 30.59593954341244 |
| 4 | 3 | 606.447657421981 | 62.188214277840686 |
| 8 | 3 | 1153.2822637361564 | 158.80434015047186 |
| 16 | 3 | 2159.7779254046995 | 297.0808627084906 |

Captured MPK-shaped vLLM output throughput:

| Batch | Samples | Mean tok/s | Stdev tok/s |
| ----: | ------: | ---------: | ----------: |
| 1 | 3 | 153.16926887618368 | 14.68057122933764 |
| 2 | 3 | 316.10883967853846 | 25.61002580849797 |
| 4 | 3 | 629.2727994297298 | 53.427342463807086 |
| 8 | 3 | 1256.784287803509 | 98.39254277958221 |
| 16 | 3 | 2411.4020405041215 | 138.46933756501426 |

## Remaining Gaps

- Capture or explicitly waive A100 vLLM runtime validation for the current
  H200-only paper-baseline path.
- Run matching MPK and VDCores baseline rows under the same `Qwen/Qwen3-8B`
  serving policies before making cross-method paper claims.
- Compare against PTO persistent-device once the matching serving runner
  exists.
