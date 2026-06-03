# 2026-06-03 PTO serving command plan

## Code And Data Changed

- Added a `pto_persistent_device_qwen3_8b_full_serving` target-method run to
  the paper-baseline viewer data, tied to the `pto_persistent_device`
  baseline and both `mpk_offline_decode` and `vdcores_offline_decode`.
- Extended the serving command planner to emit PTO resource-backed Qwen
  diagnostic runner commands and strict full-serving viewer-import commands
  for every policy batch.
- Linked `paper_readiness_work_item_001` to the generated PTO command-plan
  selectors so dispatcher backlog validation can catch stale execution plans.

## Architecture Quality

- Keeps the repo-owned PTO target method in the same reviewable command-plan
  path as MPK, VDCores, vLLM, SGLang, and ThunderKittens rows.
- Preserves the full-serving import gate: the importer command is reviewable,
  but it remains blocked until raw PTO rows contain model-equivalent decode
  correctness plus latency and throughput metrics.

## Evaluation Run

- RED: updated the review-artifact tests to require the missing PTO
  command-plan rows and work-queue selectors; the focused tests failed while
  the plan still emitted 36 records and the PTO selectors were empty.
- GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_serving_command_plan_generates_policy_commands \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  -q
```

Result: `2 passed in 0.19s`.

## Remaining Gaps

The PTO command plan is now reviewable, but full-serving promotion still
requires raw PTO MPK-policy and VDCores-policy rows with model-equivalent
decode correctness plus latency and throughput metrics.
