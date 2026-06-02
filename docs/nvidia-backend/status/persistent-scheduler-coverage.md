# CUDA Backend Status: Persistent Scheduler Coverage

## Persistent Scheduler Coverage

The CUDA persistent-device scheduler has verified coverage for the mechanics
needed by current review artifacts. It is still distinct from full normal PTO
graph coverage, which remains tracked separately under remaining gaps.

Verified scheduler mechanics include:

- descriptor-driven DAG launch through the persistent-device runtime;
- device-side scheduler blocks releasing dependent work through completion
  accounting and bounded completion-ring paths;
- configurable scheduler blocks, worker blocks, worker blocks per task,
  `stream_id`, `block_dim`, and queue capacity in paired A100/H200 smokes;
- repeat-run lifecycle reset for direct, queue, DAG-chain, graph-descriptor,
  scratch-reuse, and tensor-core persistent scenarios;
- selected benchmark rows for graph spellings such as reordered, named
  callable, role-map, submit-groups, parallel-chain, wide-fanout,
  multi-fan-in, and layered-cross descriptors;
- stable report labels for known scheduler errors, including
  `unreachable_task`.

Representative evidence lives in the persistent-scheduler remaining-gap
archive because those files were originally written while closing the gap:

- [Part 1](remaining-gaps/persistent-scheduler-generalization/part-01.md)
  records node-link, node-port, task-dictionary, submits, and submit-group
  descriptor smoke evidence.
- [Part 2](remaining-gaps/persistent-scheduler-generalization/part-02.md)
  records named-callable and selected benchmark matrix evidence.
- [Part 3](remaining-gaps/persistent-scheduler-generalization/part-03.md)
  records lifecycle, resource-policy, scheduler-scaling, graph-family, and
  error-taxonomy evidence.

The benchmark viewer also exposes these claims as structured coverage data in
`docs/nvidia-backend/benchmark-viewer/data/persistent_scheduler_coverage.json`.
That record is validated against implementation and documentation symbols by
the benchmark-viewer data guard.

The verified coverage above is enough for current CUDA backend review claims
about scheduler mechanics. The remaining open work is normal PTO graph breadth
and additional scheduler-negative coverage, not the scheduler launch,
resource, lifecycle, or artifact-reporting mechanics already captured here.
