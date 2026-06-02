# CUDA Backend Status: Persistent Scheduler Generalization

The persistent-device scheduler has verified launch, resource-policy,
lifecycle, graph-descriptor, scheduler-scaling, and artifact-reporting
coverage. See
[persistent scheduler coverage](../../persistent-scheduler-coverage.md) for
the implemented-and-verified summary.

## Open Gap

The remaining backend gap is normal PTO graph breadth, not the core persistent
scheduler mechanics. Before this gap can be closed, the CUDA backend still
needs:

- normal PTO task graph construction beyond the current scene-test
  `persistent_dag_normal_graph_f32` graph-config input;
- normal PTO graph construction feeding the persistent scheduler outside
  scene-test graph config.

The current scheduler-negative taxonomy is covered for the review scope:
unsupported `func_id`, invalid dependent IDs, dependent range, fan-in
underflow, duplicate dependent, self dependent, initial fan-in, no-root, and
unreachable-task diagnostics. Future malformed normal-graph cases belong with
the normal PTO graph lowering work above rather than a separate scheduler
mechanics blocker.

## Current Evidence

Structured coverage lives in
`docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`
and is validated by
`.agents/checks/benchmark_viewer_validation/persistent_scheduler_coverage.py`.
That coverage now requires a normal-graph lowering-boundary group tied to
`simpler_setup/cuda_normal_graph.py`, the persistent-device scene-test adapter,
the `persistent_dag_normal_graph_f32` scene-test builder, the no-torch
persistent smoke, and the Qwen unit-math live example.
Current paired smoke artifacts include
`tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/`
and
`tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/`.
Normal-graph boundary evidence also includes
`tmp/cuda-backend/persistent-graph_descriptor_submits-repeat2-smoke-ea9dec01/`,
which passed paired A100/H200 smoke with `graph_lowering=normal_graph` and
validated the submitted normal graph fan-in/dependent arrays. These artifacts
prove scheduler mechanics, selected descriptor spellings, and one normal-graph
edge-lowering path, not full normal PTO graph construction from the backend
builder.
The normal graph smoke shape matrix now also supports fork-join, chain,
multi-fan-in, and layered-cross inputs through
`.agents/skills/cuda-backend-eval/scripts/cuda_persistent_smoke_impl/normal_graph_shapes.py`.
The paired artifact
`tmp/cuda-backend/persistent-normal_graph_multi_fanin-repeat2-smoke-05ed941a/`
proves the multi-fan-in normal graph path on A100 and H200 with repeat-run
lifecycle validation.
The remaining shape artifacts complete the paired normal-graph evidence matrix:
`tmp/cuda-backend/persistent-normal_graph_fork_join-repeat2-smoke-4c68620e/`,
`tmp/cuda-backend/persistent-normal_graph_chain-repeat2-smoke-4c68620e/`, and
`tmp/cuda-backend/persistent-normal_graph_layered_cross-repeat2-smoke-4c68620e/`.
Those artifacts validate `graph_lowering=normal_graph`, dispatch sequences,
fan-in/dependent arrays, and repeat-run completion on both A100 and H200.

## Promotion Gate

Close this gap only after normal PTO task graphs construct full
persistent-device scheduler inputs beyond scene-test graph config and the
resulting backend-builder path runs through the paired A100/H200 smoke or
benchmark harness. The normal-graph shape evidence above no longer relies on
descriptor spellings, but it is still a smoke harness input rather than full
normal PTO graph construction.

## Next Actions

- Add normal PTO graph lowering into the CUDA persistent-device builder.
- Run the backend-builder normal PTO graph path through paired A100/H200
  evidence before removing this page from `status.md`.

## Evidence Archive

The detailed evidence remains split into reviewable chunks so reviewers can
audit what is already proven without mistaking it for the open gap.

| File | Lines | Topic |
| --- | ---: | --- |
| [part-01.md](part-01.md) | 240 | descriptor spellings |
| [part-02.md](part-02.md) | 129 | named-callable benchmark rows |
| [part-03.md](part-03.md) | 232 | open gap plus resource coverage |
