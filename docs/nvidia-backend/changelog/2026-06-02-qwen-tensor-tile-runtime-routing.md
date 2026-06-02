# Qwen Tensor Tile Runtime Routing

## Code And Data Changed

- Routed Qwen attention `16x64x128` and MLP `16x64x256` tensor descriptors to
  Qwen-specific persistent-device WMMA task function ids.
- Kept the diagnostic `16x16x16` tensor-core path on generic `func_id=10`.
- Updated backend status and goal-progress guards so closed backend
  implementation work is separate from open paper-readiness captures.

## Architecture Quality

The routing table lives beside the Qwen tensor-tile source contract, while the
persistent smoke runner only asks for the function id matching a descriptor.
This keeps model-shape specialization explicit without changing the existing
diagnostic tensor-core benchmark contract.

## Evaluation Run

- Focused routing tests passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_cuda_benchmark_report.py -k 'qwen_model_shapes_to_specialized_task_ids or tensor_core_tile_dag_shape_uses_block_wide_wmma_task'`.
- Qwen tensor-tile manifest test passed:
  `.venv/bin/python -m pytest -q tests/ut/py/test_nvidia_qwen_task_body_math.py -k tensor_tile_source_contract`.
- Python compile and benchmark-viewer data validation passed for this slice.

## Remaining Gaps

Paper-grade tensor evidence still needs multi-repeat A100/H200 PTO rows for
the Qwen model-shape tiles, plus comparable cuBLAS Graph, CUTLASS, Triton, and
ThunderKittens rows.
