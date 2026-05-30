# 2026-05-31 Benchmark Viewer Contract

## Code And Data Changed

- Updated the benchmark viewer to render workload input metadata from
  `run.inputs`.
- Updated the method view to render `category` and `launch_model`.
- Updated the results view to read canonical `result_records` instead of only
  the compact selected-row table.
- Extended the NVIDIA review guard and focused artifact test so viewer code
  must reference the required schema fields.

## Architecture Quality

The human-facing viewer now follows the same data contract as the machine
validator. Reviewers can inspect the benchmark shape, dtype, repeat policy,
runtime category, launch model, correctness status, statistic sample count,
and raw artifact path without opening JSON by hand.

The selected-row table remains in the data file for compact summaries, but the
review tab now uses `result_records` as the canonical result surface.

## Evaluation Run

Expected verification for this report:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q

.venv/bin/python .agents/checks/validate_benchmark_viewer_data.py

node --check docs/nvidia-backend/benchmark-viewer/viewer.js

git diff --check
```

## Remaining Gaps

- The viewer still displays current compact snapshot rows. Future result
  importers must populate richer statistics before paper-grade claims can be
  made.
- Raw artifacts remain under `tmp/` and are not committed; reviewers need the
  local source notes and artifact directories when reproducing the snapshot.
