# 2026-06-02 Normal Graph Lowering Adapter

## Code And Data Changed

- Added `.agents/skills/cuda-backend-eval/scripts/cuda_normal_graph.py` with a
  small normal-graph lowering boundary for CUDA `persistent_device` smoke
  paths.
- Routed `graph_descriptor_submits` through that helper so normal graph node
  keys and `depends_on` edges materialize as the persistent DAG fan-in array,
  flattened dependent array, and per-task dependent spans.
- Added one no-GPU unit test for the lowering arrays and tagged the existing
  CUDA smoke result with `graph_lowering: normal_graph`.

## Architecture Quality

The new helper is separate from the already large persistent smoke runner. It
keeps graph-edge lowering independent from `ctypes` ABI construction, while the
smoke runner still owns the CUDA task-record factory.

## Evaluation Run

Focused verification passed:

- `pytest -q tests/ut/py/test_cuda_backend.py -k normal_graph_lowering`
- `python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
  --mode dag --dag-shape graph_descriptor_submits --task-count 3 --n 4096 \
  --queue-capacity 2 --arch compute_80`

## Remaining Gaps

This is the first no-torch normal graph lowering adapter, not complete PTO graph
construction. The remaining backend work is to connect normal `SceneTestCase`
PTO graph objects to this lowering boundary across the descriptor families.
