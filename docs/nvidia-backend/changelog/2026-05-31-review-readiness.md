# 2026-05-31 Review Readiness

## Code And Data Changed

- Kept `cuda/host_schedule` as the host-scheduled runtime for CUDA async API
  launch experiments.
- Kept `cuda/persistent_device` as the CUDA runtime that compensates for the
  missing Ascend-style AICPU by compiling the scheduler into the device binary.
- Added explicit review guard checks under `.agents/checks/` so benchmark
  docs, viewer data, examples, and evidence references stay synchronized.
- Added CUDA examples under `examples/cuda/` that wrap the evaluated smoke
  paths instead of introducing a second example framework.

## Architecture Quality

The branch separates stable review entry points from historical evaluation
logs. Current docs, examples, viewer data, and guards now form a compact path
for reviewers to inspect the NVIDIA backend without reading the full status
archive.

## Documentation

- Replaced the long `evaluation*.md` pages with short current review entry
  points.
- Moved older evaluation narrative to `docs/nvidia-backend/history/`.
- Added a static benchmark viewer under `docs/nvidia-backend/benchmark-viewer/`
  backed by committed JSON files.
- Added `.agents/` policies for NVIDIA review evidence and remote evaluation
  fallback behavior.

## Evaluation Run

- Current full paired capture: `743709f3`, `1350` samples.
- Current compact selected gate: `743709f3`, `108` samples.
- Raw artifacts remain under `tmp/cuda-backend/`.
- Source papers and extracted text remain under `tmp/sources/`.

The review guard is:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

The focused pytest wrapper is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

## Remaining Gaps

- The current captures are review gates, not final paper-grade results.
  Future slices still need richer statistics and paper-baseline reproduction.
