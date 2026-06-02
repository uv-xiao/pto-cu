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

- full graph construction from normal PTO task graphs rather than curated
  graph-descriptor tracer bullets;
- broader graph-lowering coverage beyond the current
  `persistent_dag_graph_f32` descriptor adapter and selected descriptor
  spellings.

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
Current paired smoke artifacts include
`tmp/cuda-backend/persistent-graph_descriptor_diamond-repeat2-smoke-072e396c/`
and
`tmp/cuda-backend/persistent-graph_descriptor_generic_args4-repeat2-smoke-11db2c9d/`.
Those prove scheduler mechanics and selected descriptor spellings, not normal
PTO graph construction.

## Promotion Gate

Close this gap only after normal PTO task graphs lower into the
persistent-device scheduler path, the lowered graphs run through the paired
A100/H200 smoke or benchmark harness, and viewer data records the resulting
raw `tmp/` artifacts without relying on curated descriptor-only inputs.

## Next Actions

- Add normal PTO graph lowering into the CUDA persistent-device builder.
- Run paired A100/H200 evidence for at least fork-join, chain, fan-in, and
  layered-cross shapes through the normal graph path.
- Import or reference those raw artifacts in the benchmark viewer before
  removing this page from `status.md`.

## Evidence Archive

The detailed evidence remains split into reviewable chunks so reviewers can
audit what is already proven without mistaking it for the open gap.

| File | Lines | Topic |
| --- | ---: | --- |
| [part-01.md](part-01.md) | 240 | descriptor spellings |
| [part-02.md](part-02.md) | 129 | named-callable benchmark rows |
| [part-03.md](part-03.md) | 232 | open gap plus resource coverage |
