# 2026-06-01 SGLang Fixed-Range H200 Capture

## Code And Data Changed

- Corrected the generated SGLang serving command plan to use
  `--random-range-ratio 1.0` for fixed prompt/decode lengths.
- Added SGLang H200 batch-1 online-serving and offline-engine result rows to
  the benchmark viewer for the VDCores `128/64` serving policy.
- Added a newer SGLang execution-attempt record that keeps `bench_one_batch`
  as the remaining run blocker instead of marking the full run imported.

## Architecture Quality

The source diagnosis is now explicit: SGLang
`compute_random_lens(full_len, range_ratio, num)` samples in
`[max(full_len * range_ratio, 1), full_len]`, so `range_ratio=0` is variable
length rather than fixed length. The generated command plan now matches the
review contract used by the imported result rows.

`bench_one_batch` remains separated from the online/offline result rows because
it still fails before emitting a latency row. This prevents the viewer from
claiming a complete SGLang paper baseline while still making the passing
online and offline evidence reviewable.

## Evaluation Run

The H200 checkout used the standalone pto-cu tree-sync fallback. No upstream
repository was edited or pushed.

Raw artifact:

- `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-fixedrange-bfc1c581/`

Verification commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/validate_nvidia_changelog.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/checks/check_nvidia_review_ready.py
node --check docs/nvidia-backend/benchmark-viewer/viewer.js
```

Captured online serving metrics:

| Metric | Value |
| ------ | ----: |
| completed requests | 1 |
| failed requests | 0 |
| input tokens | 128 |
| output tokens | 64 |
| request throughput | 2.48693394676766 req/s |
| output throughput | 159.16377259313023 tok/s |
| mean E2E latency | 384.5958118326962 ms |
| mean TTFT | 36.877373699098825 ms |
| mean ITL | 5.6076819948371375 ms |

Captured offline-engine metrics:

| Metric | Value |
| ------ | ----: |
| successful requests | 1 |
| input tokens | 128 |
| output tokens | 64 |
| total latency | 0.4909931207075715 s |
| request throughput | 2.036688413391408 req/s |
| output throughput | 130.34805845705012 tok/s |

## Remaining Gaps

- Resolve the SGLang `bench_one_batch` `input_ids` `None` failure.
- Capture SGLang batch `2`, `4`, `8`, and `16` under the same fixed policy.
- Add repeated samples before treating SGLang as paper-ready.
