# 2026-06-02 Normal Graph Shape Matrix Smoke

## Code And Data Changed

- Added a small normal-graph shape module for fork-join, chain, multi-fan-in,
  and layered-cross CUDA persistent-device smokes.
- Wired those shapes into the standalone and paired persistent smoke runners.
- Updated persistent scheduler coverage/status docs with the paired
  `normal_graph_multi_fanin` A100/H200 artifact.

## Architecture Quality

The shape definitions live in
`.agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke_impl/`
instead of adding another large hand-built DAG block to the main smoke runner.
Each shape lowers through `lower_normal_graph`, then uses the existing
persistent scheduler ABI and validator contracts.

## Evaluation Run

- Red check failed before implementation because `normal_graph_multi_fanin`
  was not an accepted `cuda_persistent_smoke.py --dag-shape` choice.
- Passed local A100 smokes for `normal_graph_fork_join`,
  `normal_graph_chain`, `normal_graph_multi_fanin`, and
  `normal_graph_layered_cross`.
- Passed paired A100/H200 smoke with tree-sync fallback:
  `tmp/cuda-backend/persistent-normal_graph_multi_fanin-repeat2-smoke-05ed941a/`.
- Passed paired validation for `graph_lowering=normal_graph`,
  fan-in `0,0,0,3`, dependents `3,3,3`, dispatch `1,2,11,6`,
  scalar args `scalar0=2.0`, tensor args `c=tmp2`, and repeat completions
  `4,4`.

## Remaining Gaps

- Normal PTO task-graph construction beyond scene-test graph config remains
  open.
- Fork-join, chain, and layered-cross normal graph shapes had local A100
  smoke evidence in this change. The later
  [normal graph paired shape completion](2026-06-02-normal-graph-paired-shape-completion.md)
  report adds the paired A100/H200 rows.
