# 2026-06-02 C++ Orchestrator Snapshot Smoke

## What Changed

- Added `normal_graph_cpp_orchestrator_chain` to the CUDA persistent smoke
  shapes.
- Added `make_cpp_orchestrator_snapshot_submits` under the persistent smoke
  normal-graph shape module. The smoke path builds a live C++ `Orchestrator`,
  submits real `TaskArgs`, snapshots NEXT_LEVEL slots, and lowers the snapshot
  through `simpler_setup/cuda_pto_graph.py`.
- Extended paired-smoke expectations for dispatch, fan-in/dependent arrays,
  graph task args, and `graph_lowering=normal_graph`.

## Architecture Quality

The new smoke shape keeps C++ snapshot construction in the normal-graph shape
module instead of expanding the large smoke runner. The persistent runtime still
receives the same scheduler arrays, but the source path is now explicit in the
artifact as `graph_source=cpp_orchestrator_snapshot`.

## Evaluation

Focused local checks passed:

```bash
.venv/bin/python -m pytest -q tests/ut/py/test_cuda_backend.py \
  -k 'cpp_snapshot_normal_graph_shape or reads_live_cpp_orchestrator'
```

Local A100 smoke passed:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke.py \
  --device 0 --task-count 3 --n 1024 --arch compute_80 --mode dag \
  --queue-capacity 2 --dag-shape normal_graph_cpp_orchestrator_chain \
  --repeat-runs 2 \
  --output-json tmp/cuda-backend/cpp-orchestrator-snapshot-working/a100.json
```

Paired A100/H200 smoke passed after refreshing the H200 editable build:

```bash
PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
  .agents/skills/cuda-backend-eval/scripts/cuda_pair_persistent_smoke.py \
  --branch goal/nvidia-paper-ready \
  --output-root tmp/cuda-backend/cpp-orchestrator-snapshot-paired-working \
  --dag-shape normal_graph_cpp_orchestrator_chain \
  --task-count 3 --queue-capacity 2 --repeat-runs 2 \
  --skip-remote-refresh
```

Artifacts:

- `tmp/cuda-backend/cpp-orchestrator-snapshot-working/a100.json`
- `tmp/cuda-backend/cpp-orchestrator-snapshot-paired-working/`
  `persistent-normal_graph_cpp_orchestrator_chain-repeat2-smoke-8513e1f5/`

The paired validator required dispatch `[1,1,1]`, fan-in `[0,1,1]`,
dependents `[1,2]`, `graph_task_arg_key=cpp_orchestrator_snapshot`,
`graph_lowering=normal_graph`, repeat completions `[3,3]`, and zero scheduler
errors on both GPUs.
