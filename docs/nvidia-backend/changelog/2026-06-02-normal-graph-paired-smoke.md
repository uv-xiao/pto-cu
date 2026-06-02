# 2026-06-02 Normal Graph Paired Smoke

## Code And Data Changed

- Added `--expected-graph-lowering` to the CUDA smoke validator.
- Made the paired persistent smoke command require
  `graph_lowering=normal_graph` for `graph_descriptor_submits`.
- Updated persistent scheduler status and benchmark-viewer coverage data with
  the paired A100/H200 normal-graph smoke evidence.

## Architecture Quality

The review artifact now checks the CUDA normal-graph boundary explicitly
instead of inferring it from fan-in and dependent arrays alone. The paired
smoke still targets the existing persistent scheduler ABI, so this change
adds evidence without adding a new runtime path.

## Evaluation Run

- Red check before implementation used `cuda_validate_smoke.py` on
  `tmp/cuda-backend/normal-graph-validator-red/a100.json` with
  `--expected-graph-lowering normal_graph` and
  failed because the validator did not accept `--expected-graph-lowering`.
- Passed local A100 smoke:
  `.agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py`
  produced
  `tmp/cuda-backend/normal-graph-submits-local/a100.json`.
- Passed paired A100/H200 smoke with tree-sync fallback, avoiding remote Git
  fetch:
  `tmp/cuda-backend/persistent-graph_descriptor_submits-repeat2-smoke-ea9dec01/`.
- Passed paired validation:
  `.agents/skills/cuda-backend-eval/scripts/cuda_validate_smoke.py`
  checked both artifacts with `--expected-graph-lowering normal_graph`.

## Remaining Gaps

- The remaining persistent scheduler generalization gap stays open for normal
  PTO task-graph construction beyond scene-test graph config and broader
  paired A100/H200 normal-graph shapes.
