# 2026-05-31 A100 Runtime Launch Viewer Evidence

## Code And Data Changed

- Added a `direct_runtime` CUDA benchmark path. It builds a small nvcc shared
  library and measures its CUDA Runtime API `cudaLaunchKernel` vector-add
  launch path.
- Added a `direct_runtime` benchmark-viewer method, capture-import mapping,
  and A100 result record.
- Updated the host-schedule paper-evaluation matrix and regenerated
  `paper_readiness_audit.json`.
- Hardened the CUDA benchmark unit tests, NVIDIA review guard, and focused
  review artifact test for the new Runtime API baseline.

## Architecture Quality

The host-schedule launch-overhead claim now distinguishes:

- PTO host-scheduled CUDA async launch;
- CUDA Runtime API `cudaLaunchKernel`;
- CUDA Driver API `cuLaunchKernel`;
- CUDA Driver graph replay.

The Runtime API baseline is not derived from a Driver module handle. It uses an
nvcc-built shared library so the measured path is genuinely owned by CUDA
Runtime API calls.

## Evaluation Run

The imported raw capture contains ten passing A100 repeats:

```bash
PYTHONPATH=$PWD:$PWD/python CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024 --repeats 10 --block-dim 256 \
    --arch compute_80 --label host-launch-runtime-a100-e429c07b \
    --output-dir tmp/cuda-backend/host-launch-runtime-a100-e429c07b
```

The imported `direct_runtime` record has `sample_count=10`,
`host_wall_ns=267274`, `device_wall_ns=252415`, and `correctness=pass`.

## Remaining Gaps

- The capture covers A100 only.
- The capture covers the vector host-schedule shape only.
- The host-schedule launch-overhead claim still needs H200 Runtime/Driver rows,
  selected tensor launch shapes, and full p50/p90/p99 statistics before it can
  be promoted to paper-ready.
