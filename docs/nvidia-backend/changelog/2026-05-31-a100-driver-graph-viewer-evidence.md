# 2026-05-31 A100 Driver Graph Viewer Evidence

## Code And Data Changed

- Added a `direct_driver_graph` mapping to
  `docs/nvidia-backend/benchmark-viewer/data/capture_imports.json`.
- Exported `tmp/cuda-backend/host-launch-a100-8b6cdaee/cuda-benchmark.json`
  into viewer-compatible result records.
- Imported the A100 `host_schedule_vector_ops` `direct_driver_graph` row into
  `docs/nvidia-backend/benchmark-viewer/data/results.json`.
- Updated the host-schedule paper-evaluation matrix and regenerated the
  readiness audit so the claim points at concrete A100 Driver graph evidence.
- Hardened the focused review test and NVIDIA review guard so the new import
  mapping and result record remain reviewable.

## Architecture Quality

The host-schedule launch-overhead claim now has an explicit same-work CUDA
Driver graph row derived from raw `tmp/` artifacts instead of only naming the
baseline as future work. The viewer still marks the claim as not paper-ready,
but the blockers are narrower and concrete: direct Runtime API rows, H200 graph
rows, tensor launch shapes, and full distribution captures.

## Evaluation Run

The imported raw capture contains ten passing A100 repeats:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_viewer_export.py \
    tmp/cuda-backend/host-launch-a100-8b6cdaee/cuda-benchmark.json \
    --artifact-root tmp/cuda-backend/host-launch-a100-8b6cdaee/ \
    --output tmp/cuda-backend/host-launch-a100-8b6cdaee/viewer-result-records.json
```

The imported `direct_driver_graph` record has `sample_count=10`,
`host_wall_ns=31927`, `device_wall_ns=17920`, and `correctness=pass`.

## Remaining Gaps

- The capture covers A100 only.
- The capture covers the vector host-schedule shape only.
- The claim still needs direct CUDA Runtime API rows, H200 graph rows, selected
  tensor launch shapes, and full p50/p90/p99 statistics before it can be
  promoted to paper-ready.
