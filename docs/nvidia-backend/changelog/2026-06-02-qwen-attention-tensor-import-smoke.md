# Qwen Attention Tensor Import Smoke

## Code And Data Changed

- Captured a local A100 one-repeat Qwen attention tensor tile smoke under
  `tmp/cuda-backend/qwen-attention-tensor-target-bdce5ea4/`.
- Added guarded `import_smoke` metadata to the Qwen attention model-shape
  target in tensor workload coverage data.
- Updated the benchmark viewer to render import-smoke status, artifact root,
  and scope for model-shape tensor targets.

## Architecture Quality

The model-shape tensor plan now has executable import evidence. The viewer
mapping can route PTO persistent-device and cuBLAS Graph rows for the
16x64x128 Qwen attention tile without promoting a one-repeat smoke to
paper-grade throughput.

## Evaluation Run

- Local A100 single-repeat PTO persistent-device tensor-core and cuBLAS Graph
  captures passed.
- CUDA viewer export passed and produced two `tensor_core_tile` records from
  the local smoke.
- Focused benchmark-viewer validation passed with the new import-smoke guard.

## Remaining Gaps

The smoke is not final paper-grade data. The open work remains tuned PTO
tensor bodies and multi-repeat A100/H200 comparisons against cuBLAS Graph,
CUTLASS, Triton, and ThunderKittens.
