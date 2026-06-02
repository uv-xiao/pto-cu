# Tensor Import Smoke Record Guard

## Code And Data Changed

- Tightened tensor workload coverage validation so each import-smoke method
  record must match the tensor-core benchmark id, hardware, compute target,
  tensor shape, raw artifact path, correctness status, and sample count.
- Added explicit `A100` and `compute_80` metadata to the Qwen attention and
  MLP tensor import-smoke records.

## Architecture Quality

The review data now proves that imported viewer rows describe the intended
model-shape target and artifact, instead of only proving that a method name
and sample count were present.

## Evaluation Run

- Focused benchmark-viewer data validation passed after the stricter
  exported-record checks.
- Focused changelog, review-ready, viewer schema, and diff checks passed.

## Remaining Gaps

The guarded records are still one-repeat local A100 import smokes. Paper-grade
rows still require multi-repeat A100/H200 captures and tuned PTO tensor bodies.
