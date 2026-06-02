# 2026-06-02 Qwen Policy-Length Execution Contract

## Code And Data Changed

- Added `policy_length_complete` to resource-backed decode-step execution
  summaries.
- Added `qwen_resource_backed_policy_length_decode_execution` when every
  selected serving policy executes its full planned decode-step count.
- Updated PTO paper-readiness queue text to keep policy-length diagnostic
  execution separate from full Qwen numerical correctness and full-serving
  row import.

## Architecture Quality

The execution result now distinguishes a short bounded smoke from a diagnostic
run that covers the full MPK and VDCores decode-step policies. This keeps the
review signal precise without weakening the full-serving promotion guard.

## Evaluation Run

Verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_single_context_session.py \
  tests/ut/py/test_nvidia_qwen_decode_loop_runner.py \
  tests/ut/py/test_nvidia_review_artifacts.py -q
```

The benchmark-viewer data, changelog, CUDA example, and NVIDIA review guards
also passed.

## Remaining Gaps

PTO full-serving rows still require full Qwen numerical correctness and
full-serving viewer result import for MPK and VDCores policy rows.
