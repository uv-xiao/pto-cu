# 2026-05-31 A100 Driver Launch Viewer Evidence

## Code And Data Changed

- Added a `direct_driver` method entry for the raw CUDA Driver API
  `cuLaunchKernel` baseline.
- Added a `direct_driver` mapping to
  `docs/nvidia-backend/benchmark-viewer/data/capture_imports.json`.
- Exported `tmp/cuda-backend/host-launch-a100-8b6cdaee/cuda-benchmark.json`
  and imported the A100 `direct_driver` row into `results.json`.
- Updated the host-schedule paper-evaluation matrix and regenerated the
  readiness audit so A100 raw Driver launch evidence is explicit.
- Hardened the focused review test and NVIDIA review guard for the new method,
  import rule, and result row.

## Architecture Quality

The host-schedule launch-overhead claim now separates three concepts:

- PTO `host_schedule` launch behavior;
- raw CUDA Driver API launch through `cuLaunchKernel`;
- CUDA Driver graph replay.

This avoids treating Driver API evidence as CUDA Runtime API evidence. The
matrix still keeps the true Runtime API row as an open paper-readiness gap.

## Evaluation Run

The imported raw capture contains ten passing A100 repeats:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py \
    tmp/cuda-backend/host-launch-a100-8b6cdaee/cuda-benchmark.json \
    --artifact-root tmp/cuda-backend/host-launch-a100-8b6cdaee/ \
    --output tmp/cuda-backend/host-launch-a100-8b6cdaee/viewer-result-records.json
```

The imported `direct_driver` record has `sample_count=10`,
`host_wall_ns=44934`, `device_wall_ns=30719`, and `correctness=pass`.

## Remaining Gaps

- The capture covers A100 only.
- The capture covers the vector host-schedule shape only.
- The claim still needs direct CUDA Runtime API rows, H200 Driver launch and
  graph rows, selected tensor launch shapes, and full p50/p90/p99 statistics
  before it can be promoted to paper-ready.
