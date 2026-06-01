# 2026-06-01 vLLM H200 Serving Sweep

## Code And Data Changed

- Imported H200 vLLM `Qwen/Qwen3-8B` serving rows for VDCores-comparable
  batch/concurrency `2`, `4`, `8`, and `16`.
- Added a vLLM execution-attempt record that groups the sweep raw artifacts
  and keeps the overall vLLM paper-baseline run marked partial.
- Preserved the raw server command, server readiness response, benchmark
  commands, benchmark logs, status files, and JSON outputs under `tmp/`.

## Architecture Quality

The serving viewer now represents the full VDCores-comparable vLLM batch sweep
instead of a single batch-1 point. The execution attempt remains partial
because the vLLM paper-baseline run also covers the MPK-comparable 1024-token
serving policy and repeated samples for paper variance.

This keeps the data model honest: completed rows are imported into
`results.json`, while incomplete paper requirements stay visible in the work
queue through the partial execution-attempt blocker.

## Evaluation Run

The H200 checkout was refreshed through the documented tree-sync fallback.
Repository Actions stayed disabled, and no upstream repository was edited or
pushed.

Raw artifact:

- `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-vdcores-qwen3-8b-sweep-89fe1705/`

Serving results:

| Batch | Completed | Failed | Mean TTFT ms | Mean ITL ms | Output tokens/s |
| ----- | --------- | ------ | ------------ | ----------- | --------------- |
| 2 | 2 | 0 | 80.22279106080532 | 5.732923454146773 | 287.79827100668354 |
| 4 | 4 | 0 | 34.042275743559 | 5.801673921283394 | 635.7423030915141 |
| 8 | 8 | 0 | 82.2809441597201 | 5.931017967495357 | 1108.0923559038586 |
| 16 | 16 | 0 | 116.38721390045248 | 5.672932683309127 | 2119.4372078699585 |

Verification command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result:

```text
benchmark viewer data validation passed
```

## Remaining Gaps

- Capture the MPK-comparable 1024-token vLLM serving policy.
- Repeat vLLM serving samples for variance and paper confidence intervals.
- Capture the matching MPK serving path and import it into the viewer.
- Compare the same serving workload against PTO persistent-device once that
  runner is available.
