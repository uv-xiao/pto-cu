# CUDA Backend Status: Kernel Compiler Coverage

## Kernel Compiler Coverage

CUDA kernel compiler integration has verified coverage for current
host-schedule and persistent-device review artifacts. The previous
scene-test argument builder breadth item is now treated as verified for the
current review scope, not as an open backend blocker.

Verified coverage includes:

- host-schedule task-body compilation with `nvcc`-produced PTX;
- generated persistent-device dispatch tables for selected task bodies;
- role-aware graph descriptor lowering into CUDA `TaskArgs` metadata;
- explicit graph-descriptor variants for tagged in/out, role-keyed inputs,
  generic arguments, triad, quad, tensor-core tile, diamond, and scratch-reuse
  DAG shapes;
- real-data `SceneTestCase` paths for current CUDA host-schedule and
  persistent-device tracer bullets;
- paired A100/H200 smoke and selected benchmark rows with source-paper
  provenance, sanitized reconstruction commands, Markdown/SVG reports, and
  zero scheduler errors for selected generated-dispatch rows.

The benchmark viewer also exposes the current scene-builder coverage as
`evaluations/nvidia/benchmark-viewer/data/scene_builder_coverage.json`, so
reviewers can inspect covered builders, open work, and exact code/doc evidence
without searching the large CUDA scene-test file manually.

Representative evidence remains in the kernel-compiler integration archive
because those files were written while closing the integration gap:

- [Part 1](remaining-gaps/kernel-compiler-integration/part-01.md) starts the
  original kernel compiler integration status.
- [Part 2a](remaining-gaps/kernel-compiler-integration/part-02a.md) and
  [Part 2b](remaining-gaps/kernel-compiler-integration/part-02b.md) document
  host-schedule task-body compilation, persistent generated dispatch, graph
  node metadata, and role lowering.
- [Parts 3-8](remaining-gaps/kernel-compiler-integration/part-03.md) document
  graph-descriptor generated dispatch, real-data scene paths, and paired
  A100/H200 smoke evidence.
- [Parts 9-10](remaining-gaps/kernel-compiler-integration/part-09.md) document
  selected benchmark promotion, target-specific PTX, diamond and scratch-reuse
  descriptor evidence, and the builder-breadth closure path.

The verified coverage above is enough for current review claims about CUDA
task body compilation and generated persistent dispatch. Future model-specific
argument layouts can be added with new CUDA examples, but they are not required
to close the current backend implementation status item.
