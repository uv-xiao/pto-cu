# CUDA Backend Status: Tuned Tensor Workloads

## Open Gap

The tensor DAG rows validate descriptor metadata, generated dispatch,
multi-fragment WMMA descriptors, explicit graph tensor descriptors, and
library/generated baseline comparisons. Current viewer data includes A100 and
H200 tensor rows for PTO persistent tensor paths, cuBLAS Graph, CUTLASS, and
Triton. The benchmark report includes throughput tables and SVGs, so
scheduler-vs-task work separation exists for current diagnostic tensor bodies.

Structured review evidence lives in
`docs/nvidia-backend/benchmark-viewer/data/tensor_workload_coverage.json` and
is checked against current viewer result records by the benchmark-viewer data
guard. The same viewer data now lists the first model-shape target tiles that
must receive tuned PTO rows. Capture import rules are now tensor-tile aware,
so future model-shape captures can be imported without colliding with the
existing 16x16x16 diagnostic rows. The first Qwen attention and MLP targets
also have local A100 one-repeat import smokes under
`tmp/cuda-backend/qwen-attention-tensor-target-bdce5ea4/` and
`tmp/cuda-backend/qwen-mlp-tensor-target-c1110223/`, proving that PTO
persistent-device and cuBLAS Graph rows route through the viewer mapping for
the 16x64x128 and 16x64x256 tiles.

The Qwen tensor-tile source contract is ready in
`examples/cuda/qwen_persistent_task_bodies_impl/tensor_tiles.py`: it defines
block-wide WMMA task functions for the attention `16x64x128` tile and MLP
`16x64x256` tile, with fixed-shape guards and `m16n16k8` TF32/F32 fragment
metadata. The benchmark/runtime descriptors now route those two Qwen
model-shape descriptors to Qwen-specific function ids instead of the generic
diagnostic WMMA task. The remaining gap is paper-readiness throughput capture,
not backend descriptor-shape, first tensor-core callable plumbing, baseline
viewer coverage, source-contract definition, or runtime selector wiring.

Needed:

- broader model-kernel shape families once the tensor-core/library path
  exists;
- multi-repeat throughput rows that compare tuned PTO tensor bodies against
  cuBLAS Graph, CUTLASS, Triton, and ThunderKittens baselines.

## Current Evidence

Structured evidence lives in
`docs/nvidia-backend/benchmark-viewer/data/tensor_workload_coverage.json` and
is validated by
`.agents/checks/benchmark_viewer_validation/tensor_workload_coverage.py`.
The current one-repeat model-shape import smokes are under
`tmp/cuda-backend/qwen-attention-tensor-target-bdce5ea4/` and
`tmp/cuda-backend/qwen-mlp-tensor-target-c1110223/`.
The source-contract evidence is emitted by
`examples/cuda/qwen_persistent_task_bodies.py` as
`qwen_tensor_tile_contract`.

## Promotion Gate

Close this paper-evaluation gap only after the Qwen tensor-tile function ids
produce multi-repeat A100 and H200 rows for the Qwen attention and MLP target
tiles, import through the viewer with correctness and throughput statistics,
and keep baseline rows for cuBLAS Graph, CUTLASS, Triton, and ThunderKittens
comparable.

## Next Actions

- Capture multi-repeat A100/H200 PTO and baseline rows for those tiles.
- Import the raw `tmp/` artifacts into the viewer and update this archived
  page once the paper-readiness work queue is complete.
