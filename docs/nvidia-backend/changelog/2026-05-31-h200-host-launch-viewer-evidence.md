# 2026-05-31 H200 Host Launch Viewer Evidence

## Code And Data Changed

- Imported a 10-repeat H200 host-launch capture into benchmark-viewer
  `result_records`.
- Added H200 viewer evidence for `direct_runtime`, `direct_driver`, and
  `direct_driver_graph` on the same `n=1024` vector shape as the PTO
  host-schedule row.
- Updated the host-schedule paper-evaluation matrix and regenerated
  `paper_readiness_audit.json`.
- Changed shared capture-import wording so host-launch Runtime and Driver rows
  are no longer described as A100-only.

## Architecture Quality

The host-schedule launch-overhead claim now has a cross-GPU vector-launch
comparison:

- PTO host-scheduled CUDA async launch on A100 and H200;
- CUDA Runtime API `cudaLaunchKernel` on A100 and H200;
- CUDA Driver API `cuLaunchKernel` on A100 and H200;
- CUDA Driver graph replay on A100 and H200.

This is still a microbenchmark result, not a paper-ready claim. The remaining
work is tensor-shape launch coverage plus distribution statistics that expose
p50, p90, p99, mean, standard deviation, and throughput.

## Evaluation Run

The H200 run used the remote tree-sync fallback instead of remote Git refresh:

```bash
rsync -a --delete --exclude=.venv --exclude=build --exclude=tmp \
  --exclude=__pycache__ --exclude=.pytest_cache \
  "$PWD/" bizhaoh200:/data/shibizhao/pto-cu/
```

The imported raw capture contains ten passing H200 repeats:

```bash
CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/skills/cuda-backend-eval/scripts/cuda_benchmark.py \
    --device 0 --sizes 1024 --repeats 10 --block-dim 256 \
    --arch compute_90 --label host-launch-h200-ec8f272e \
    --output-dir tmp/cuda-backend/host-launch-h200-ec8f272e
```

Imported H200 medians:

| Method | Sample Count | Host Wall Ns | Device Wall Ns |
| --- | ---: | ---: | ---: |
| PTO host-schedule | 10 | 25340 | 14976 |
| CUDA Runtime API | 10 | 157191 | 146336 |
| CUDA Driver API | 10 | 32077 | 20911 |
| CUDA Driver graph | 10 | 26296 | 14991 |

## Remaining Gaps

- The capture covers the vector host-schedule shape only.
- The host-schedule launch-overhead claim still needs selected tensor launch
  shapes and full p50/p90/p99 statistics before it can be promoted to
  paper-ready.
