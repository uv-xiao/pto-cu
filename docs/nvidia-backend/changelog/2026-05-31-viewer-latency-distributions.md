# 2026-05-31 Viewer Latency Distributions

## Code And Data Changed

- Extended `cuda_viewer_export.py` so repeated CUDA captures export p50, p90,
  p99, mean, standard deviation, minimum, and maximum latency fields for both
  host wall time and device wall time.
- Regenerated committed CUDA viewer result records from the current raw
  artifacts under `tmp/cuda-backend/`.
- Updated the benchmark viewer table to show host and device p90 columns when
  distribution fields are available.
- Tightened the benchmark-viewer validator so repeated
  `median_capture_group` records must include the distribution fields.

## Architecture Quality

The viewer no longer hides repeated-capture shape behind a single median.
`host_wall_ns` and `device_wall_ns` remain p50 aliases for existing renderer and
test compatibility, while the explicit distribution fields expose outliers to
reviewers. For example, the H200 Runtime API row now shows a much larger p90
than p50, making launch variance visible in the review data instead of buried
inside raw JSON.

## Evaluation Run

The TDD red test was:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest \
  tests/ut/py/test_nvidia_review_artifacts.py::test_cuda_viewer_export_generates_contract_records \
  -q
```

It failed because exported records lacked `host_wall_p50_ns`. After the
exporter change, the same test passed.

The committed viewer records were regenerated with `cuda_viewer_export.py`
from these raw artifacts:

- `tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/`
- `tmp/cuda-backend/host-launch-a100-8b6cdaee/`
- `tmp/cuda-backend/host-launch-runtime-a100-e429c07b/`
- `tmp/cuda-backend/host-launch-h200-ec8f272e/`

## Remaining Gaps

- Single-sample compact rows naturally have degenerate distribution fields.
- Distribution fields now exist for imported repeated captures, but the
  host-schedule launch-overhead claim still needs selected tensor launch
  shapes and actual stream-count or graph-replay sweep captures.
