# Sharded Result Records Restored

## Code And Data Changed

- Replaced the monolithic benchmark-viewer `results.json` with
  `data/results/index.json`, `record_files.json`, and one short file per
  result row under `data/results/records/`.
- Updated the HTML viewer config so it loads the sharded result manifest.
- Updated `viewer_data_io.write_json` so future imports through the logical
  `results.json` path keep writing the sharded collection instead of
  recreating the monolith.

## Architecture Quality

The viewer still exposes the same logical `result_records` collection, while
future evaluation imports should only touch the affected row files. This keeps
raw artifacts under `tmp/` and keeps committed benchmark data reviewable.

## Evaluation Run

Commands:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_qwen_unit_math_live.py -q
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_benchmark_viewer_data.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_cuda_examples.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/check_nvidia_review_ready.py
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/checks/validate_nvidia_changelog.py
```

Result: all passed.

## Remaining Gaps

This is a reviewability fix only. The next implementation work should continue
closing real Qwen persistent-device math and serving-evaluation gaps.
