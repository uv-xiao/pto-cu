# 2026-06-01 vLLM H200 MPK Serving Sweep

## Code And Data Changed

- Imported the H200 vLLM `Qwen/Qwen3-8B` MPK-comparable serving sweep for
  batch/concurrency `2`, `4`, `8`, and `16`.
- Added one execution-attempt record for the sweep and four benchmark-viewer
  result rows. The raw server and benchmark artifacts remain under `tmp/`.
- Kept the full vLLM paper-baseline run partial because repeated samples,
  matching MPK serving evidence, and PTO persistent-device comparison are not
  complete.

## Architecture Quality

The viewer now has the full single-sample vLLM serving shape matrix for both
tracked decode policies: VDCores-comparable `128 -> 64` and MPK-comparable
`64 -> 1024`, each at batch/concurrency `1`, `2`, `4`, `8`, and `16`. The
new MPK rows use the same `paper_baseline_serving_capture` schema as the
previous vLLM rows, so the viewer can compare TTFT, ITL, throughput, request
shape, and raw artifact location without special handling.

## Evaluation Run

The H200 checkout was refreshed through the documented tree-sync fallback.
Repository Actions stayed disabled, and no upstream repository was edited or
pushed.

Raw artifact:

- `tmp/cuda-backend/paper-baselines/serving-runs/vllm/h200-mpk-qwen3-8b-sweep-908438de/`

Serving results:

| Batch | Completed | Failed | Mean TTFT (ms) | Mean ITL (ms) | Output tok/s |
| ----: | --------: | -----: | -------------: | ------------: | -----------: |
| 2 | 2 | 0 | 72.44588294997811 | 5.906464019926849 | 334.7536743804337 |
| 4 | 4 | 0 | 39.24347530119121 | 5.97836579881674 | 665.049038750595 |
| 8 | 8 | 0 | 57.89269332308322 | 6.042956362181649 | 1311.7874753053661 |
| 16 | 16 | 0 | 98.91026074183173 | 5.985480906558565 | 2629.5086844579146 |

Verification command:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
```

Result:

```text
wrote docs/nvidia-backend/benchmark-viewer/data/paper_baseline_environment_plans.json
wrote docs/nvidia-backend/benchmark-viewer/data/paper_baseline_run_readiness.json
wrote docs/nvidia-backend/benchmark-viewer/data/paper_readiness_audit.json
wrote docs/nvidia-backend/benchmark-viewer/data/paper_readiness_work_queue.json
wrote docs/nvidia-backend/benchmark-viewer/data/goal_progress.json
```

## Remaining Gaps

- Repeat vLLM serving samples for variance and paper confidence intervals.
- Capture the matching MPK serving path and import it into the viewer.
- Compare the same serving workload against PTO persistent-device once that
  runner is available.
