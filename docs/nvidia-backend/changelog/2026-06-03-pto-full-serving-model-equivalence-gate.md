# 2026-06-03 PTO Full-Serving Model-Equivalence Gate

## Code And Data Changed

- Tightened the PTO Qwen full-serving importer so raw rows must declare
  `model_equivalent_ready=true` and
  `comparison_scope=model_equivalent_decode` in `correctness_details`.
- Tightened benchmark-viewer validation, paper-readiness audit logic, and the
  PTO serving preflight row gate with the same model-equivalence requirement.
- Updated the CUDA evaluation workflow, Qwen full-serving gap report, and
  paper-evaluation matrix evidence symbols.

## Architecture Quality

The full-serving row path now rejects diagnostic decode-position comparisons
even when they report `token_match=true`. This keeps the paper-readiness gate
aligned with the current Qwen blocker: prompt prefill and decode state must be
model-equivalent before MPK or VDCores rows can be imported as full-serving
evidence.

## Evaluation Run

RED:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_pto_full_serving_viewer_import.py::test_full_serving_importer_rejects_diagnostic_comparison_scope \
  tests/ut/py/test_nvidia_benchmark_viewer_result_validation.py::test_viewer_data_validator_rejects_diagnostic_comparison_scope \
  tests/ut/py/test_nvidia_paper_readiness_audit.py::test_paper_readiness_rejects_diagnostic_comparison_scope \
  -q
```

Result before the gate: `3 failed`; the importer, viewer validator, and paper
audit accepted diagnostic comparison scope.

GREEN:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_pto_full_serving_viewer_import.py \
  tests/ut/py/test_nvidia_benchmark_viewer_result_validation.py \
  tests/ut/py/test_nvidia_paper_readiness_audit.py \
  -q
```

Result: `21 passed`.

Additional preflight row-gate check:

```bash
PYTHONPATH=$PWD:$PWD/python:$PWD/examples/cuda \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_pto_full_serving_row_gate_rejects_diagnostic_comparison_scope \
  tests/ut/py/test_nvidia_review_artifacts.py::test_pto_full_serving_row_gate_accepts_both_policy_rows \
  -q
```

Result: `2 passed`.

## Remaining Gaps

This is a guardrail, not a Qwen correctness fix. PTO still needs
model-equivalent prompt prefill, KV-cache state, Hugging Face token/logit
agreement, and MPK/VDCores policy rows with latency and throughput metrics.
