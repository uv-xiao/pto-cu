# Legacy Next Evaluation Gaps

## Next Evaluation Gaps

- Add model-shaped kernels and more repetitions before treating any
  worker-grid setting as a tuned baseline.
- Replace the scalar GEMM body with a CUDA implementation closer to the
  intended tensor-core/tiling backend once the runtime ABI can carry richer
  tensor metadata.
