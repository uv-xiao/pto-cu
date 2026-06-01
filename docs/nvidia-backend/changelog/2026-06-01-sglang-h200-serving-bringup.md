# 2026-06-01 SGLang H200 Serving Bring-Up

## Code And Data Changed

- Captured H200 SGLang serving bring-up attempts from the isolated
  `sglang-7ed53d15` environment.
- Added a benchmark-viewer execution-attempt record for the successful online
  serving run and kept it out of result rows because the measured token shape
  did not match the planned VDCores policy.
- Refreshed the generated paper-readiness audit, work queue, run readiness,
  environment plans, and goal-progress data.

## Architecture Quality

The SGLang path now records command-level failure and retry evidence instead of
conflating bring-up with paper results. The attempt captures three important
runtime facts for future runs:

- SGLang's default piecewise CUDA graph path hit an illegal memory access on
  Qwen3-8B/H200, while `--disable-piecewise-cuda-graph` let the server reach
  readiness.
- Offline mode needs either `random-ids` or a local dataset path; otherwise the
  `random` dataset path tries to fetch a ShareGPT helper file.
- The successful online run still reported `38` input and `44` output tokens
  for a requested `128/64` policy, so the evidence must remain an execution
  attempt until the SGLang token-shape contract is corrected.

## Evaluation Run

The H200 checkout used the standalone pto-cu tree-sync fallback. No upstream
repository was edited or pushed.

Raw artifacts:

- `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-8a61669c/`
- `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-disablepwcg-8a61669c/`
- `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-localdata-8a61669c/`
- `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-randomids-8a61669c/`

Captured online metrics from the final random-ids run:

| Metric | Value |
| ------ | ----: |
| completed requests | 1 |
| failed requests | 0 |
| measured input tokens | 38 |
| measured output tokens | 44 |
| request throughput | 0.9697372581994511 req/s |
| output throughput | 42.66843936077585 tok/s |
| mean E2E latency | 1011.1441072076559 ms |
| mean TTFT | 771.7542587779462 ms |
| mean ITL | 5.566300629356572 ms |

Verification command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

## Remaining Gaps

- Make SGLang `bench_serving` preserve the VDCores `128/64` token policy before
  importing result rows.
- Resolve the `bench_offline_throughput` context-length/tokenization failure.
- Resolve the `bench_one_batch` `input_ids` `None` failure.
- Capture batch `1`, `2`, `4`, `8`, and `16` after batch `1` is
  policy-shaped, then add repeated samples for confidence intervals.
