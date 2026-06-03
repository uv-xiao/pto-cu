# 2026-06-03 Work Queue Execution Selectors

## Code And Data Changed

- Updated the paper-readiness work-queue generator so next actions that omit
  `serving_workload_ids` inherit the referenced paper-baseline run's serving
  workloads.
- Regenerated review artifacts so the VDCores execution-attempt blocker now
  carries `vdcores_qwen3_8b_decode_preflight:vdcores_offline_decode`.
- Updated the dispatcher backlog to show that selector on
  `paper_readiness_work_item_004`.

## Architecture Quality

Execution-attempt blockers now remain tied to the same reviewable command-plan
surface as matrix-missing evidence. This makes the VDCores shared-instruction
blocker actionable without requiring dispatchers to infer the serving policy
from a separate run record.

## Evaluation Run

RED:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  -q
```

The focused test failed while the execution-attempt item had empty
`serving_workload_ids` and no serving command selectors.

GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  -q
```

Result: `1 passed`.

## Remaining Gaps

The selector makes the VDCores blocker dispatchable, but the actual baseline
still needs a runnable segmented or token-windowed shared-instruction schedule
before full-serving rows can be imported.
