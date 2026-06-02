# CUDA Backend Status: Persistent Scheduler Generalization

The persistent-device scheduler has verified launch, resource-policy,
lifecycle, graph-descriptor, scheduler-scaling, and artifact-reporting
coverage. See
[persistent scheduler coverage](../../persistent-scheduler-coverage.md) for
the implemented-and-verified summary.

## Review-Scope Closure

The previous backend gap was normal PTO graph breadth, not the core persistent
scheduler mechanics. For the current review scope, this page is closed:
malformed normal-graph lowering cases for live C++ snapshot inputs now have
focused coverage.

The current scheduler-negative taxonomy is covered for the review scope:
unsupported `func_id`, invalid dependent IDs, dependent range, fan-in
underflow, duplicate dependent, self dependent, initial fan-in, no-root,
unreachable-task diagnostics, and malformed C++ snapshot-input lowering.
Future malformed normal-graph cases belong with normal PTO graph lowering
rather than a separate scheduler mechanics blocker.

## Current Evidence

Structured coverage lives in
`evaluations/nvidia/benchmark-viewer/data/persistent_scheduler_coverage.json`
and is validated by
`.agents/checks/benchmark_viewer_validation/persistent_scheduler_coverage.py`.
That coverage now requires a normal-graph lowering-boundary group tied to
`simpler_setup/cuda_pto_graph.py`, `simpler_setup/cuda_normal_graph.py`, the
persistent-device scene-test adapter, the `persistent_dag_normal_graph_f32`
scene-test builder, the no-torch persistent smoke, and the Qwen unit-math
live example.
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
The C++ orchestrator now exposes live NEXT_LEVEL task-slot snapshots before
drain resets the ring state. The CUDA PTO graph adapter converts those
snapshots, real `TaskArgs` plus `TensorArgType` tags, and Python orchestration
`submit_next_level` calls into persistent-device submit records. The Qwen
unit-math live example lowers PTO-style tagged submit records through
`simpler_setup/cuda_pto_graph.py`. These prove builder-side dependency
inference outside scene-test graph config and include live C++ task-slot
capture.
The paired artifact
`tmp/cuda-backend/cpp-orchestrator-snapshot-paired-working/`
`persistent-normal_graph_cpp_orchestrator_chain-repeat2-smoke-8513e1f5/`
now validates a live C++ orchestrator snapshot input on A100 and H200. It
requires `dag_shape=normal_graph_cpp_orchestrator_chain`,
`graph_source=cpp_orchestrator_snapshot`, `graph_lowering=normal_graph`,
dispatch `[1,1,1]`, fan-in `[0,1,1]`, dependents `[1,2]`, repeat completions
`[3,3]`, and zero scheduler errors.
Malformed normal-graph lowering cases for that same live C++ snapshot input
path are covered by
`tmp/cuda-backend/cpp-orchestrator-snapshot-malformed-326cb61/`
`snapshot-malformed.json`. The artifact requires expected failures for
multi-`TaskArgs` snapshot entries, tensor-name arity mismatch, and duplicate
snapshot slot keys.

## Promotion Gate

This page can stay out of `status.md` remaining gaps while the evidence above
stays current. Reopen it only if a new normal PTO graph construction path
outgrows the current live-snapshot malformed coverage. The normal-graph shape
evidence no longer relies on descriptor spellings, and the paired C++ snapshot
smoke plus `TaskArgs`/tagged-submit adapter proves the builder-side dependency
rule.

## Next Actions

- Keep malformed normal-graph lowering cases with
  `cuda_snapshot_malformed.py` as new snapshot-input failure modes are added.

## Evidence Archive

The detailed evidence remains split into reviewable chunks so reviewers can
audit what was proven while closing the gap.

| File | Lines | Topic |
| --- | ---: | --- |
| [part-01.md](part-01.md) | 240 | descriptor spellings |
| [part-02.md](part-02.md) | 129 | named-callable benchmark rows |
| [part-03.md](part-03.md) | 232 | open gap plus resource coverage |
