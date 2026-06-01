# 2026-06-02 Qwen Resource-Backed Viewer Import

## Code And Data Changed

- Added `pto_qwen_resource_backed_viewer_import.py` to convert the
  resource-backed diagnostic runner artifact into benchmark-viewer rows.
- Imported two A100 rows, one for `mpk_offline_decode` and one for
  `vdcores_offline_decode`.
- Updated viewer validation to recognize
  `diagnostic_resource_backed_qwen_dag` as diagnostic, not full-serving,
  coverage.

## Architecture Quality

The raw resource-backed execution evidence is now visible in the HTML viewer's
normal result table instead of only as a raw-artifact link. The row shape names
include the serving policy, and the coverage label keeps the result out of the
full-serving paper-ready bucket until numerically correct Qwen kernels and
full-serving imports exist.

## Evaluation Run

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/\
pto_qwen_resource_backed_viewer_import.py \
  tmp/cuda-backend/pto-serving-resource-backed-execution-2026-06-02/\
qwen-decode-loop-runner.json
```

Result: `results.json` now contains two
`diagnostic_resource_backed_qwen_dag` rows. Both rows record
`completed_count=255`, `error_count=0`, and `correctness=pass`.

Focused verification:

```bash
PYTHONPATH=$PWD:$PWD/python \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_resource_backed_viewer_import.py -q
```

Result: `2 passed`.

## Remaining Gaps

- Promote PTO only after full-serving Qwen rows with numerically correct task
  bodies are imported.
- Continue the VDCores and ThunderKittens full-serving baseline work.
