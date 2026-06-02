# Qwen MLP Tensor Import Smoke

## Code And Data Changed

- Captured a local A100 one-repeat Qwen MLP tensor tile smoke under
  `tmp/cuda-backend/qwen-mlp-tensor-target-c1110223/`.
- Added guarded `import_smoke` metadata to the Qwen MLP model-shape target.
- Tightened tensor workload coverage validation so the two initial Qwen
  model-shape targets must both carry import-smoke evidence.

## Architecture Quality

Both initial model-shape tensor targets now have executable import evidence.
The data path proves PTO persistent-device and cuBLAS Graph records can be
captured and routed separately for the 16x64x128 attention tile and the
16x64x256 MLP tile.

## Evaluation Run

- Local A100 single-repeat PTO persistent-device tensor-core and cuBLAS Graph
  MLP-tile captures passed.
- CUDA viewer export passed and produced two `tensor_core_tile` records from
  the local MLP smoke.
- Focused benchmark-viewer validation passed with the stricter two-smoke
  target guard.

## Remaining Gaps

The smokes are not final paper-grade data. The open work remains tuned PTO
tensor bodies and multi-repeat A100/H200 comparisons against cuBLAS Graph,
CUTLASS, Triton, and ThunderKittens.
