# 2026-06-01 Qwen Unit Math Viewer Import

## Code And Data Changed

- Added `pto_qwen_unit_math_viewer_import.py` to normalize the live Qwen
  unit-math artifact into a benchmark-viewer result row.
- Imported
  `tmp/cuda-backend/pto-serving-unit-math-live-2026-06-01/qwen-unit-math-live.json`
  as a `llm_serving_decode` PTO persistent-device diagnostic row.
- Refreshed the paper-evaluation matrix, readiness audit, and work queue so
  reviewers can see that unit-math viewer evidence is present but not
  full-serving evidence.

## Architecture Quality

The live unit-math artifact proved RMSNorm, QKV cache writeback, SwiGLU, and
logits on A100, but it was still hidden in `tmp/`. This import makes that
evidence visible in the HTML benchmark viewer while keeping
`statistic.serving_coverage=diagnostic_unit_math`.

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_unit_math_live.py::test_unit_math_live_importer_marks_result_as_diagnostic -q

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/pto_qwen_unit_math_viewer_import.py \
  tmp/cuda-backend/pto-serving-unit-math-live-2026-06-01/qwen-unit-math-live.json

PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/refresh_nvidia_review_artifacts.py
```

## Remaining Gaps

- Execute the full Qwen decode loop.
- Import PTO full-serving `Qwen/Qwen3-8B` viewer rows.
