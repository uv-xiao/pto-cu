# CUDA Current Evaluation Capture

The current review baseline is the paired A100/H200 capture from commit
`743709f3`. It adds the nine-task layered-cross graph descriptor to both the
full benchmark matrix and the compact selected gate.

## Captures

| Capture | Artifact root | Samples | Purpose |
| --- | --- | ---: | --- |
| Full paired matrix | `tmp/cuda-backend/current-head-full-layered-cross-fixed/combined-current-743709f3/` | 1350 | Three sizes, three repeats, selected host, persistent, CUDA Graph, cuBLAS, tensor, and graph rows. |
| Compact paired gate | `tmp/cuda-backend/layered-cross-selected-current-fixed/combined-current-743709f3/` | 108 | Review-size gate for selected baselines and graph descriptor rows. |
| Stream concurrency | `tmp/cuda-backend/combined-stream-pool6-02bca4df/` | 40 | Ten-repeat A100/H200 comparison between serial host-schedule launches and parallel launches on separate CUDA streams. |
| Graph replay sweep | `tmp/cuda-backend/graph-replay-sweep-01e30e99/` | 120 | Ten-repeat A100/H200 CUDA Driver graph replay sweep for vector sizes 1024, 4096, 65536 and matching naive SGEMM graph replay. |
| Direct launch sweep | `tmp/cuda-backend/direct-launch-sweep-626b8c75/` | 240 | Ten-repeat A100/H200 CUDA Runtime and CUDA Driver direct-launch sweep for the same vector sizes and naive SGEMM tile shape. |
| SGLang fixed-range serving | `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-batch1-fixedrange-bfc1c581/` | 2 result rows | H200 Qwen3-8B batch-1 online serving and offline engine throughput for the VDCores `128/64` policy. |
| SGLang fixed-range serving sweep | `tmp/cuda-backend/paper-baselines/serving-runs/sglang/h200-vdcores-qwen3-8b-fixedrange-sweep-cfbdcf0c/` | 8 result rows | H200 Qwen3-8B batch `2`, `4`, `8`, and `16` online serving and offline engine throughput for the VDCores `128/64` policy. |

These captures passed `cuda_validate_capture.py` with the matching current
presets during the latest verification run.

## Runtime Rows

| Row | Runtime meaning | Review signal |
| --- | --- | --- |
| `pto_host_schedule` | Host launches CUDA callables asynchronously through the CUDA host runtime. | Phase-1 launch path and stream API surface. |
| `direct_runtime` | CUDA Runtime API launches vector and naive SGEMM kernels directly from host code. | Runtime API launch baseline for host-schedule comparisons. |
| `direct_driver` | CUDA Driver API launches vector and naive SGEMM kernels directly from host code. | Driver API launch baseline for host-schedule comparisons. |
| `direct_driver_graph` | Raw CUDA Driver API graph replay. | Host-launch amortization baseline. |
| `pto_persistent_device` | Persistent CUDA kernel executes task descriptors on worker blocks. | First device-side scheduler baseline. |
| `pto_persistent_dag_graph_layered_cross` | Persistent scheduler executes a nine-task explicit graph descriptor. | Current graph topology and lifetime-stress gate. |
| `cublas_sgemm_graph` | cuBLAS SGEMM captured and replayed through a CUDA Graph. | Library-backed tensor baseline. |

## Layered-Cross Gate

The layered-cross descriptor has nine tasks with dispatch ids
`1,2,11,1,2,1,6,1,1`. Its graph fan-in array is
`0,0,0,2,3,1,2,3,2`, and its dependent list is
`3,3,4,4,5,4,6,7,6,7,7,8,8`.

The benchmark rows record `scalar0=2.0` and tensor alias `c=a`, which verifies
that descriptor metadata is still visible in generated reports and validators.

## Viewer

Open [benchmark-viewer/index.html](benchmark-viewer/index.html) through a
local static file server to inspect the benchmark setups, method definitions,
result snapshot, paper-baseline execution attempts, and commands:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/docs/nvidia-backend/benchmark-viewer/`.

## History

Older captures are archived under [history/index.md](history/index.md). New
evaluation updates should add a focused history entry there instead of growing
this current page.
