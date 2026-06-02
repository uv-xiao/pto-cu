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
`evaluations/nvidia/benchmark-viewer/data/persistent_scheduler_coverage.json`.
That record is validated against implementation and documentation symbols by
the benchmark-viewer data guard. The same coverage data now requires a
normal-graph lowering-boundary group, with evidence tied to
`simpler_setup/cuda_pto_graph.py`, `simpler_setup/cuda_normal_graph.py`, the
`persistent_dag_normal_graph_f32` scene-test builder in
`simpler_setup/scene_test.py`, the no-torch persistent smoke, and the Qwen
unit-math CUDA example. The C++ orchestrator now exposes live NEXT_LEVEL slot
snapshots before drain/reset, the CUDA adapter converts those snapshots and
real `TaskArgs` plus `TensorArgType` tags into CUDA submit records, and the
Qwen example constructs the persistent DAG from PTO-style tagged submits.
Together these provide concrete builder-side evidence outside scene-test graph
config.

The verified coverage above is enough for current CUDA backend review claims
about scheduler mechanics and scene-test normal graph construction. Paired
A100/H200 smoke now also validates the submit-chain normal graph boundary with
`graph_lowering=normal_graph`, and the normal graph smoke matrix now covers
fork-join, chain, multi-fan-in, and layered-cross shape construction with
paired A100/H200 evidence for each shape. Paired A100/H200 smoke also
validates `normal_graph_cpp_orchestrator_chain` with
`graph_source=cpp_orchestrator_snapshot`, dispatch `[1,1,1]`, fan-in
`[0,1,1]`, dependents `[1,2]`, and repeat-run completion on both GPUs.
Malformed live-snapshot input coverage now records expected failures for
multi-`TaskArgs` snapshot entries, tensor-name arity mismatch, and duplicate
snapshot slot keys. Persistent scheduler generalization is no longer listed as
a remaining backend gap in `status.md`; tuned tensor workloads remain tracked
separately.
