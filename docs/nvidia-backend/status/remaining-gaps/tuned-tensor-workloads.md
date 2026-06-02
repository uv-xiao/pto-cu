# CUDA Backend Status: Tuned Tensor Workloads

## Tuned Tensor Workloads

The tensor DAG rows validate descriptor metadata, generated dispatch,
multi-fragment WMMA descriptors, explicit graph tensor descriptors, and
library/generated baseline comparisons. Current viewer data includes A100 and
H200 tensor rows for PTO persistent tensor paths, cuBLAS Graph, CUTLASS, and
Triton. The benchmark report includes throughput tables and SVGs, so
scheduler-vs-task work separation exists for current diagnostic tensor bodies.

Structured review evidence lives in
`docs/nvidia-backend/benchmark-viewer/data/tensor_workload_coverage.json` and
is checked against current viewer result records by the benchmark-viewer data
guard. The remaining gap is tuned PTO tensor body work at model-relevant
sizes, not descriptor-shape, first tensor-core callable plumbing, or baseline
viewer coverage.

Needed:

- tuned PTO tensor body implementation beyond the current diagnostic WMMA and
  scalar tiled-GEMM rows;
- broader model-kernel shape families once the tensor-core/library path
  exists;
- multi-repeat throughput rows that compare tuned PTO tensor bodies against
  cuBLAS Graph, CUTLASS, Triton, and ThunderKittens baselines.
