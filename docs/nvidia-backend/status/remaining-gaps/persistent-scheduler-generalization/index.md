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
  spellings;
- additional scheduler-negative cases beyond the current labeled error
  taxonomy for unsupported `func_id`, invalid dependent IDs, dependent range,
  fan-in underflow, duplicate dependent, self dependent, initial fan-in, and
  no-root or unreachable-task diagnostics.

## Evidence Archive

The detailed evidence remains split into reviewable chunks so reviewers can
audit what is already proven without mistaking it for the open gap.

| File | Lines | Topic |
| --- | ---: | --- |
| [part-01.md](part-01.md) | 240 | descriptor spellings |
| [part-02.md](part-02.md) | 129 | named-callable benchmark rows |
| [part-03.md](part-03.md) | 232 | open gap plus resource coverage |
