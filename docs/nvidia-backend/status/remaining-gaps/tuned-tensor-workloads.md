# CUDA Backend Status: Tuned Tensor Workloads

## Tuned Tensor Workloads

The tensor DAG row validates descriptor metadata and generated dispatch, but
the GEMM body is a scalar microbenchmark rather than a tuned tensor-core
kernel. The first paired tensor-shape sweep now covers `8x4x12`,
`16x16x64`, and `32x16x64` descriptors on A100 and H200. The first
`tensor_core_tile` smoke and selected-baseline benchmark row now validate a
WMMA generated-dispatch task body on both GPUs for one-fragment and
multi-fragment descriptors. The benchmark report includes a signed
DAG-increment table and SVG, so scheduler-vs-task work separation exists for
current microbenchmarks. The remaining gap is tuned tensor execution and
comparative throughput at model-relevant sizes, not descriptor-shape or first
tensor-core callable plumbing.

Needed:

- tensor-core or library-backed callable body tuning beyond the current
  small multi-fragment WMMA benchmark row;
- broader model-kernel shape families once the tensor-core/library path
  exists;
- real tuned-kernel throughput rows beyond the current scheduler-adjusted
  microbenchmark deltas.

