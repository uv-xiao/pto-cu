# 2026-05-31 Tensor Launch Viewer Evidence

## Code And Data Changed

- Added direct CUDA Driver, CUDA Runtime, and CUDA Driver Graph naive SGEMM
  baselines to `cuda_benchmark.py`.
- Extended paired benchmark validation, capture presets, capture-import rules,
  benchmark descriptions, method descriptions, and focused tests for the new
  tensor-launch baseline IDs.
- Imported A100 and H200 `tensor_core_tile` viewer records for
  `direct_driver`, `direct_runtime`, and `direct_driver_graph`.
- Regenerated `paper_readiness_audit.json` after removing the selected tensor
  launch-shape blocker from the host-schedule launch-overhead claim.

## Architecture Quality

The new tensor rows use the same column-major `16x16x16` descriptor contract as
the existing cuBLAS and persistent tensor rows. The Driver and Driver Graph
paths compile a PTX kernel with nvcc and load it through the CUDA Driver API;
the Runtime path compiles the same shape into the existing nvcc shared-library
baseline. Remote tree-sync runs now set `PTO_SOURCE_COMMIT` so raw H200
artifacts record the synced source commit instead of the stale remote Git
checkout commit.

## Evaluation Run

The TDD red tests first failed because `run_single_sample` rejected
`direct_driver_sgemm`, `direct_runtime_sgemm`, and
`direct_driver_graph_sgemm`. After implementation, the focused dispatch tests
passed.

Local A100 smoke checks passed for all three new single-baseline modes. The
10-repeat artifacts imported into the viewer are:

- `tmp/cuda-backend/tensor-launch-a100-09462d04/`
- `tmp/cuda-backend/tensor-launch-h200-09462d04/`

Both raw captures validated with 110 rows, including the three direct tensor
baselines and `16x16x16` tensor-tile metadata.

## Remaining Gaps

- The host-schedule launch-overhead claim still needs actual stream-count and
  graph-replay sweeps across selected vector and tensor shapes.
- The tensor-core baseline claim still needs CUTLASS or CuTe and broader
  ThunderKittens rows before it is paper-ready.
