# 2026-06-01 Qwen Proxy Decode Loop Viewer Import

## Code And Data Changed

- Added `pto_qwen_microdecode_viewer_import.py` to import live PTO Qwen proxy
  microdecode-loop artifacts into benchmark-viewer result records.
- Imported the A100 repeated decode-loop artifact as
  `serving_coverage=diagnostic_microdecode`.
- Tightened PTO serving preflight so diagnostic Qwen rows cannot satisfy the
  full-serving row check.

## Architecture Quality

The viewer now shows the repeated proxy decode-loop evidence as a structured
result row instead of only a raw artifact link. The result is explicitly
diagnostic, which preserves the paper-readiness boundary between controlled
proxy execution and full Qwen/Qwen3-8B serving.

## Evaluation Run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_microdecode_loop.py -q
```

Result: passed after the import and preflight filter updates.

## Remaining Gaps

- PTO still lacks numerically correct Qwen kernels and a full-serving
  `serving_coverage=full_serving` viewer result row.
- The imported row is useful diagnostic evidence only.
