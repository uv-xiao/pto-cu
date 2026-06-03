# 2026-06-03 VDCores Window Plan Validation

## Code And Data Changed

Added
`.agents/skills/cuda-backend-eval/scripts/vdcores_validate_instruction_window_plan.py`
to validate the VDCores Qwen3-8B shared-instruction window-plan artifact. The
validator checks the 512-instruction shared-table limit, contiguous compute and
memory windows, lower-bound window counts, and the required segmented-runtime
handoff contract.

`paper_baseline_execution_attempts.json` now records the validator under the
`vdcores_qwen3_8b_shared_instruction_window_plan_h200` attempt. The attempt
summary includes `window_contract_validation=pass` and keeps
`runnable_handoff_contract_status=required_not_implemented`, so the artifact is
reviewable without pretending it is a runnable VDCores paper row.

## Architecture Quality

The benchmark-viewer data guard now treats the VDCores shared-window attempt as
a special contract: the raw plan must remain `analysis_only`, every emitted
window must fit inside the shared-instruction capacity, and the summary must
keep `paper_row_importable=false`.

The work queue fallback text also carries the validator-backed handoff status,
so dispatcher reviews can see that the next missing step is builder/runtime
support rather than another analysis artifact.

## Evaluation Run

RED tests first failed on the missing validator script and missing work-queue
summary text. After implementation, the focused guard run passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_vdcores_instruction_window_plan_validator_guards_handoff_contract \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  tests/ut/py/test_nvidia_review_artifacts.py::test_ultimate_goal_artifacts_define_paper_ready_cuda_path \
  -q
```

The real artifact validator also passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/vdcores_validate_instruction_window_plan.py \
  tmp/cuda-backend/paper-baselines/vdcores/qwen3-8b-shared-window-plan-0a0392d2/shared-window-plan.json
```

## Remaining Gaps

This does not make VDCores full-serving paper-ready. The row still needs
builder/runtime support for a segmented or token-windowed
shared-instruction schedule that advances windows while preserving resident
tensors, KV-cache state, dependencies, correctness, and per-window timing.
