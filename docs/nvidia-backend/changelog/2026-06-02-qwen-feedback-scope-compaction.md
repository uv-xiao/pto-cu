# Qwen Feedback Scope Compaction

## Code And Data Changed

- Added `decode_feedback_scope=single_sequence_row0_greedy_argmax` to the
  generated `qwen_logits` task-body manifest and decode-feedback summaries.
- Updated the resource-backed Qwen viewer importer to carry that scope into
  compact benchmark rows, including older raw artifacts whose feedback status
  already proves device or host sampled-token commits.
- Kept the stronger policy-length artifact
  `tmp/cuda-backend/pto-serving-decode-loop-policy-sampled-ref-5948323b/` and
  removed two weaker duplicate MPK/VDCores diagnostic rows from viewer data.

## Architecture Quality

The diagnostic decode-loop evidence now states the exact feedback scope instead
of implying full serving or batched token sampling. This keeps the current
resource-backed implementation honest while preserving the useful policy-length
MPK and VDCores rows.

## Evaluation Run

No new live GPU run was needed for this slice. The compact viewer data was
refreshed from existing raw artifacts under `tmp/`, and the relevant rows now
show full logits-buffer checking, `unit_math_full_rmsnorm`, device-observed
token feedback, and the explicit row-zero single-sequence feedback scope.

Verification:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_task_body_math.py::test_logits_task_body_declares_single_sequence_feedback_scope \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py::test_resource_backed_importer_emits_diagnostic_rows \
  -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
```

Result: both focused tests passed, and benchmark-viewer data validation passed
after regenerating readiness data from the compacted result rows.

## Remaining Gaps

This does not close the full Qwen serving gate. The remaining blocker is still
full-serving PTO rows with end-to-end generated-token correctness for the MPK
and VDCores serving policies.
