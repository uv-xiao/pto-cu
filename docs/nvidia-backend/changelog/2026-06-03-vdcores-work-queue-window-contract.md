# 2026-06-03 VDCores Work Queue Window Contract

## Code And Data Changed

The paper-readiness work-queue generator now carries the validated VDCores
shared-window handoff contract on the `vdcores_full_serving_qwen3_8b`
matrix-missing-evidence item. The committed
`paper_readiness_work_queue.json` entry now includes
`window_contract_validation=pass`,
`runnable_handoff_contract_status=required_not_implemented`, and the
`vdcores_validate_instruction_window_plan.py` guard.

## Architecture Quality

This keeps the matrix blocker aligned with the execution-attempt evidence. A
reviewer no longer has to cross-reference the execution attempt to see that the
next VDCores step is runnable segmented-window builder/runtime support, not a
new analysis-only capacity plan.

## Evaluation Run

RED first failed on the missing VDCores window-contract text in the
matrix-missing-evidence item. After the generator and committed data update,
the focused work-queue guard passed:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_paper_readiness_work_queue_matches_current_audit \
  -q
```

## Remaining Gaps

This does not make the VDCores Qwen3-8B full-serving row runnable or
paper-ready. The backend still needs segmented or token-windowed
shared-instruction execution that preserves resident tensors, KV-cache state,
dependencies, correctness, and timing across windows.
