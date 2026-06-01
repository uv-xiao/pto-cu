# NVIDIA Backend Paper-Ready Evaluation Plan: Workloads

## Workloads

The workload ladder should grow from controlled kernels to paper-level systems:

- vector ABI workloads: add, mul, scale, square, axpy, affine, triad, quad, and
  generic argument packets;
- DAG scheduling workloads: chain, diamond, layered-cross, fan-in, fan-out, and
  queue-capacity pressure;
- tensor workloads: tiled GEMM, tensor-core tile kernels, shape sweeps, and
  stream or graph variants;
- lifecycle workloads: repeated runtime init, module load, allocation, copy,
  launch, synchronize, teardown, and rebuild separation;
- LLM-serving workloads: decode microsteps, paged KV-cache movement, attention,
  GEMM, all-reduce or all-gather when multi-GPU is in scope.

Each workload must have a benchmark viewer entry, example or script entry point,
and a raw artifact path.

