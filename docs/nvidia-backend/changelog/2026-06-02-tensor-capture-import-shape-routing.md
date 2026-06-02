# Tensor Capture Import Shape Routing

## Code And Data Changed

- Made CUDA viewer export rules filter matching capture rows by optional
  tensor tile shape.
- Added capture-import rules for the first Qwen attention and MLP tensor tile
  targets, paired with PTO persistent-device and cuBLAS Graph methods.
- Updated capture-import validation so same baseline, size, and task-count
  rules can coexist when they target different tensor tile shapes.

## Architecture Quality

The benchmark import path can now route diagnostic 16x16x16 tensor-core rows
and model-shape tensor target rows through separate review records. That keeps
the viewer data contract shape-aware without changing benchmark kernels.

## Evaluation Run

- Focused exporter tests passed with both legacy 16x16x16 rows and a
  16x64x128 model-shape row.
- Full benchmark-viewer validation passed with the new capture-import rule
  keys.

## Remaining Gaps

This change prepares the data path. It does not claim tuned PTO tensor
performance for the model-shape rows; those captures remain a backend gap.
