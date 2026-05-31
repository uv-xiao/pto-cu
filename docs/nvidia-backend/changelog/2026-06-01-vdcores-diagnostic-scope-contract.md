# 2026-06-01 VDCores Diagnostic Scope Contract

## Code And Data Changed

- Updated the persistent-device scheduler paper matrix so the remaining VDCores
  blocker asks for stable baseline instrumentation, not an impossible
  non-diagnostic queue/scheduler trace.
- Added explicit `measurement_scope` metadata to the VDCores resource-policy
  run and the imported scheduler-trace result.
- Added a focused unit guard so future edits keep diagnostic scheduler fields
  separate from final non-diagnostic latency and correctness rows.

## Architecture Quality

The VDCores row now distinguishes two evidence classes:

- final guarded RepeatM latency/correctness evidence without profile-slot
  diagnostics;
- diagnostic queue-pressure and allocwarp scheduler-lifetime evidence from the
  tmp-only profile-slot patch.

This prevents the viewer and paper-readiness queue from implying that CUDA
baseline scheduler counters can be collected without instrumentation. The
paper contract is now explicit: either VDCores grows a stable instrumentation
mode with the same semantics, or the diagnostic trace remains a separate
explanatory row.

## Evaluation Run

No remote benchmark was rerun in this slice. The existing raw artifacts remain:

- `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-guard-bench-f6b16bac/`
- `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-repeat-guard-correctness-712f88e8/`
- `tmp/cuda-backend/paper-baselines/vdcores/qwen3-1p7b-queue-scheduler-46872fa4/`

Local review commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
PYTHONPATH=$PWD:$PWD/python .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python .agents/checks/validate_nvidia_changelog.py
PYTHONPATH=$PWD:$PWD/python .agents/checks/check_nvidia_review_ready.py
```

## Remaining Gaps

- Add stable, non-ad-hoc VDCores instrumentation if scheduler/queue metrics
  must become paper-grade baseline counters.
- Keep the diagnostic queue/scheduler trace separate from non-diagnostic
  latency rows until that stable instrumentation exists.
