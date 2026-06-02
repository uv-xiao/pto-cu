# Tensor Import Smoke Commands

## Code And Data Changed

- Added rerun commands to the Qwen attention and MLP tensor import-smoke
  records.
- Extended tensor workload validation so import smokes must list the matching
  PTO, cuBLAS Graph, tensor-tile, and viewer-export commands.
- Updated the benchmark viewer to render paper-target commands separately
  from one-repeat import-smoke commands.

## Architecture Quality

Reviewers can now see the exact commands that reproduced the local A100
model-shape import smokes without confusing those commands with the planned
20-repeat paper-target captures.

## Evaluation Run

- Focused benchmark-viewer validation passed with the import-smoke command
  guard.
- Focused viewer tests passed for the JSON-backed review data.

## Remaining Gaps

The commands reproduce one-repeat local import smokes. Paper-grade tensor rows
still require multi-repeat A100/H200 captures and tuned PTO tensor bodies.
