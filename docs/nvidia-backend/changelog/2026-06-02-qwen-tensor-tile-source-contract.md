# Qwen Tensor Tile Source Contract

## Code And Data Changed

- Added `examples/cuda/qwen_persistent_task_bodies_impl/tensor_tiles.py` with
  compact source contracts for the Qwen attention `16x64x128` and MLP
  `16x64x256` WMMA tensor tiles.
- Exposed the contract through `examples/cuda/qwen_persistent_task_bodies.py`
  as `qwen_tensor_tile_contract`.
- Updated tensor workload coverage data and the remaining-gap page to narrow
  the open work to runtime selector wiring plus paired throughput capture.

## Architecture Quality

The source contract keeps Qwen model-shape tensor bodies in the existing Qwen
example package instead of growing the benchmark script. The backend status now
distinguishes source-contract readiness from the still-open runtime-wiring and
paper-throughput requirements.

## Evaluation Run

- Focused unit test passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_task_body_math.py -k tensor_tile_source_contract`.
- Python compile, benchmark-viewer data validation, CUDA example validation,
  changelog validation, and NVIDIA review guard passed for this slice.

## Remaining Gaps

The Qwen tensor-tile function ids are not yet selectable by the persistent
benchmark/runtime path, and no multi-repeat A100/H200 throughput rows have been
captured for these specialized bodies.
