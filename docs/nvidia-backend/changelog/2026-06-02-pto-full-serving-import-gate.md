# 2026-06-02 PTO Full-Serving Import Gate

## Code And Data Changed

- Tightened `pto_qwen_full_serving_viewer_import.py` so raw PTO rows must
  declare `runtime=cuda/persistent_device`.
- Required raw rows to declare `serving_coverage=full_serving` before they can
  be imported as paper-ready Qwen full-serving viewer rows.
- Added a regression test rejecting
  `diagnostic_resource_backed_qwen_dag` rows at the full-serving importer.
- Added evidence under
  `tmp/cuda-backend/pto-full-serving-import-gate-2026-06-02/`.

## Architecture Quality

The promotion path now separates diagnostic resource-backed execution from
paper-ready full-serving capture at the importer boundary. Correctness details
alone are no longer sufficient; the raw artifact must also declare the runtime
and full-serving coverage class.

## Evaluation Run

Focused verification passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_pto_full_serving_viewer_import.py -q
```

Result: `6 passed in 0.03s`.

## Remaining Gaps

This closes an import-safety gap only. Full Qwen numerical correctness and
MPK/VDCores full-serving row import remain open.
