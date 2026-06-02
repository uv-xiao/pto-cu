# 2026-06-02 Normal Graph Paired Shape Completion

## Code And Data Changed

- Corrected the paired-smoke expected dependent order for
  `normal_graph_layered_cross` to match normal graph source-order lowering.
- Updated persistent scheduler coverage and status docs to show paired
  A100/H200 evidence for fork-join, chain, multi-fan-in, and layered-cross
  normal graph shapes.

## Architecture Quality

The remaining persistent scheduler status now separates two claims cleanly:
normal graph shape breadth is covered by paired smoke artifacts, while normal
PTO task-graph construction beyond scene-test graph config remains open.

## Evaluation Run

- Passed paired A100/H200 smokes with tree-sync fallback:
  `tmp/cuda-backend/persistent-normal_graph_fork_join-repeat2-smoke-4c68620e/`,
  `tmp/cuda-backend/persistent-normal_graph_chain-repeat2-smoke-4c68620e/`,
  and
  `tmp/cuda-backend/persistent-normal_graph_layered_cross-repeat2-smoke-4c68620e/`.
- Revalidated existing paired A100/H200 evidence:
  `tmp/cuda-backend/persistent-normal_graph_multi_fanin-repeat2-smoke-05ed941a/`.
- The validator checks covered `graph_lowering=normal_graph`, dispatch
  sequences, fan-in arrays, dependent arrays, task counts, and repeat-run
  completions.

## Remaining Gaps

- Normal PTO task-graph construction beyond scene-test graph config still
  remains open.
