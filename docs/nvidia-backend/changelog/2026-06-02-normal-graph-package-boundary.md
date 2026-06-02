# 2026-06-02 Normal Graph Package Boundary

## Code And Data Changed

- Moved the CUDA normal graph lowering helper from the evaluation script tree
  into `simpler_setup/cuda_normal_graph.py`.
- Updated the persistent-device smoke runner and unit test to import the helper
  from `simpler_setup`.
- Kept the smoke output marker `graph_lowering: normal_graph` unchanged, so
  existing review artifacts can still identify the lowering path.

## Architecture Quality

Normal graph edge lowering is now package code rather than evaluator-only
script code. The helper remains independent from `ctypes` and CUDA task record
construction, while the smoke runner keeps ownership of the current ABI
materialization.

## Evaluation Run

Focused verification passed:

- `pytest -q tests/ut/py/test_cuda_backend.py -k normal_graph_lowering`
- `python .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
  --mode dag --dag-shape graph_descriptor_submits --task-count 3 --n 4096 \
  --queue-capacity 2 --arch compute_80`

## Remaining Gaps

This package boundary is still a first lowering slice. Full PTO graph object
construction into CUDA persistent-device descriptors remains open across the
broader descriptor families.
