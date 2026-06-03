# CUDA Backend Status: Tuned Tensor Workloads

## Open Gap

The tensor DAG rows validate descriptor metadata, generated dispatch,
multi-fragment WMMA descriptors, explicit graph tensor descriptors, and
library/generated baseline comparisons. Current viewer data includes A100 and
H200 tensor rows for PTO persistent tensor paths, cuBLAS Graph, CUTLASS, and
Triton. The benchmark report includes throughput tables and SVGs, so
scheduler-vs-task work separation exists for current diagnostic tensor bodies.

Structured review evidence lives in
`evaluations/nvidia/benchmark-viewer/data/tensor_workload_coverage.json` and
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

Current A100 and H200 evidence now includes three-repeat PTO
persistent-device and cuBLAS Graph captures for both Qwen model-shape tiles.
This also fixed a local capture blocker in `cuda_persistent_smoke.py`: its
`CudaPersistentDagTask` ctypes layout still used the pre-five-tensor-argument
ABI and omitted `tensor_arg_dtypes`, which made current graph tensor-core
launch packets fail with scheduler error code `5(initial_fanin_mismatch)`.
The smoke runner now matches the compiler/runtime ABI and dispatches Qwen
tensor function ids `7240` and `7241` for the target tiles.
Current generated-kernel evidence also includes A100 and H200 three-repeat
Triton and CUTLASS comparator captures for the same Qwen attention and MLP
tensor tiles. The comparator capture scripts now label records with the concrete
model-shape tile, so viewer rows for generated kernels can be checked against
the Qwen target shape instead of the generic diagnostic tensor shape.

Needed:

- broader model-kernel shape families once the tensor-core/library path
  exists;
- ThunderKittens comparator rows for the same model-shape tiles.

## Current Evidence

Structured evidence lives in
`evaluations/nvidia/benchmark-viewer/data/tensor_workload_coverage.json` and
is validated by
`.agents/checks/benchmark_viewer_validation/tensor_workload_coverage.py`.
The current one-repeat model-shape import smokes are under
`tmp/cuda-backend/qwen-attention-tensor-target-bdce5ea4/` and
`tmp/cuda-backend/qwen-mlp-tensor-target-c1110223/`.
The current A100 three-repeat tensor-throughput captures are under
`tmp/cuda-backend/qwen-attention-tensor-target-a100-repeat3-4b281f79/` and
`tmp/cuda-backend/qwen-mlp-tensor-target-a100-repeat3-4b281f79/`.
The current H200 three-repeat tensor-throughput captures are under
`tmp/cuda-backend/qwen-attention-tensor-target-h200-repeat3-85187ffd/` and
`tmp/cuda-backend/qwen-mlp-tensor-target-h200-repeat3-85187ffd/`.
The current A100 Triton/CUTLASS comparator captures are under
`tmp/cuda-backend/qwen-attention-generated-tensor-target-a100-repeat3-c743cb84/`
and
`tmp/cuda-backend/qwen-mlp-generated-tensor-target-a100-repeat3-c743cb84/`.
The current H200 Triton/CUTLASS comparator captures are under
`tmp/cuda-backend/qwen-attention-generated-tensor-target-h200-repeat3-e1ab002b/`
and
`tmp/cuda-backend/qwen-mlp-generated-tensor-target-h200-repeat3-e1ab002b/`.
The source-contract evidence is emitted by
`examples/cuda/qwen_persistent_task_bodies.py` as
`qwen_tensor_tile_contract`.

## Promotion Gate

Close this paper-evaluation gap only after the Qwen tensor-tile function ids
produce multi-repeat A100 and H200 rows for the Qwen attention and MLP target
tiles, import through the viewer with correctness and throughput statistics,
and keep baseline rows for cuBLAS Graph, CUTLASS, Triton, and ThunderKittens
comparable on the required hardware.

## Next Actions

- Capture ThunderKittens comparator rows for the same tile shapes.
- Promote the A100/H200 rows into final viewer result records only after the
  comparator set is complete.
