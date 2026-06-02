# 2026-06-02 Scene Graph Normal Lowering

## Code And Data Changed

- Routed `persistent_dag_graph_f32` scene-test graph descriptor materialization
  through `simpler_setup/cuda_normal_graph.py` for final fan-in, dependent
  array, and per-task dependent-span construction.
- Added an explicit-dependent-list entry point so scene graph descriptors keep
  user-provided dependent ordering while sharing the same fan-in/span lowering
  code.
- Kept the existing scene-test graph parsing, task-argument normalization,
  graph-level edge handling, and tensor-flow inference intact before that final
  lowering step.
- Extended `CudaNormalGraphNode` so dependency-only scene nodes can use the
  same helper as smoke nodes with concrete CUDA pointer fields.

## Architecture Quality

The normal graph lowering helper is now shared by both the no-torch smoke path
and the normal `SceneTestCase` persistent graph adapter. This moves the CUDA
backend one step closer to a single graph-to-persistent-descriptor boundary
instead of per-path hand-built fan-in arrays.

## Evaluation Run

Focused verification passed:

- `python -m py_compile simpler_setup/cuda_normal_graph.py \
  simpler_setup/scene_test.py \
  .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py`
- `pytest -q tests/ut/py/test_cuda_backend.py -k normal_graph_lowering`
- `pytest -q tests/ut/py/test_cuda_scene_test.py::\
  test_scene_test_runs_cuda_persistent_device_depends_on_graph_with_ctypes_data`
- `python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
  --mode dag --dag-shape graph_descriptor_submits --task-count 3 --n 4096 \
  --queue-capacity 2 --arch compute_80`

## Remaining Gaps

The scene-test graph adapter still constructs normalized graph descriptors from
test metadata. Full production PTO graph object construction into CUDA
persistent-device descriptors remains open.
