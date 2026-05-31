# 2026-06-01 MPK Scheduler Trace Import

## Code And Data Changed

- Imported the H200 MPK bounded-decode scheduler trace into
  `docs/nvidia-backend/benchmark-viewer/data/results.json`.
- Marked `mpk_persistent_scheduler_trace` as `imported_to_viewer` in
  `paper_baseline_runs.json`.
- Updated the persistent-device scheduler-overhead matrix to cite the MPK H200
  viewer row and removed the MPK missing-evidence blocker.
- Regenerated paper-readiness audit, work-queue, and goal-progress data.
- Added review-artifact assertions for the imported MPK scheduler row,
  scheduler-slice count, resource policy, queue-pressure policy, and generated
  kernel metadata.

Raw artifacts are under:

```text
tmp/cuda-backend/paper-baselines/mpk/
tmp/cuda-backend/paper-baselines/mpk/patched-snapshot-pointer/profile-termination-diagnostic-fa357d52/
```

## Architecture Quality

This slice turns the previous MPK profile-termination diagnostic into
reviewable benchmark-viewer evidence without changing upstream MPK. The row is
explicitly labeled as a single-sample H200 Qwen3-0.6B bounded decode with
device elapsed time from the MPK demo CUDA events and scheduler overhead from
Perfetto scheduler slice begin/end pairs.

The imported row preserves the fields needed for human review:

- 7,261 generated tasks and 1,870 task-graph events;
- 128 worker blocks and 16 local scheduler blocks on H200;
- 74,792 observed scheduler slice pairs;
- `TASK_SCHD_EVENTS` and `TASK_SCHD_PREPARE_BATCH` timing summaries;
- generated-kernel metadata for the offline MPK configuration.

No upstream repository was edited or pushed.

## Evaluation Run

The imported raw JSON is:

```text
tmp/cuda-backend/paper-baselines/mpk/persistent-scheduler-trace.json
```

It was imported with:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/paper_baseline_results_update.py \
    tmp/cuda-backend/paper-baselines/mpk/persistent-scheduler-trace.json \
    --artifact-root tmp/cuda-backend/paper-baselines/mpk/ \
    --viewer-output tmp/cuda-backend/paper-baselines/mpk/viewer-result-records.json
```

Focused verification:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_benchmark_viewer_has_json_backed_review_data \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_audit_matches_current_viewer_data \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  tests/ut/py/test_nvidia_review_artifacts.py::test_nvidia_goal_progress_matches_current_artifacts \
  -q
```

Result: `4 passed`.

The benchmark-viewer data validator also passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/validate_benchmark_viewer_data.py
```

Result: `benchmark viewer data validation passed`.

## Remaining Gaps

- The persistent-device scheduler-overhead claim still needs VDCores
  queue/resource-policy evidence before it can become paper-ready.
- The MPK row is a single bounded-decode sample, not a repeated latency sweep.
  It is enough to close the MPK scheduler-trace import blocker, but not enough
  to claim final paper-grade MPK latency distributions.
