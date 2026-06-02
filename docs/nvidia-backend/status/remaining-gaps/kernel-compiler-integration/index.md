# CUDA Backend Status: Kernel Compiler Integration

CUDA kernel compiler integration has verified host-schedule task-body
compilation, generated persistent-device dispatch, role-aware graph descriptor
lowering, real-data scene paths, and paired A100/H200 generated-dispatch
evidence. See
[kernel compiler coverage](../../kernel-compiler-coverage.md) for the
implemented-and-verified summary.

## Open Gap

The remaining backend gap is broader CUDA scene-test argument builder coverage.
Before this gap can be closed, the CUDA backend still needs argument builders
beyond the current binary elementwise, unary square, scalar scale, axpy,
affine, triad, quad, host-schedule generic args, persistent scalar/DAG tracer
bullets, and explicit graph-descriptor scratch-storage reuse paths.

## Evidence Archive

The detailed evidence remains split into reviewable chunks so reviewers can
audit what is already proven without reading every generated-dispatch artifact
as an open gap.

| File | Lines | Topic |
| --- | ---: | --- |
| [part-01.md](part-01.md) | 4 | original status |
| [part-02a.md](part-02a.md) | 222 | task-body compilation |
| [part-02b.md](part-02b.md) | 218 | role and metadata lowering |
| [part-03.md](part-03.md) | 238 | graph descriptor smoke |
| [part-04.md](part-04.md) | 227 | role-keyed graph paths |
| [part-05.md](part-05.md) | 241 | tensor-shape evidence |
| [part-06.md](part-06.md) | 242 | tensor-core descriptor |
| [part-07.md](part-07.md) | 241 | remote H200 scene paths |
| [part-08.md](part-08.md) | 238 | generic args evidence |
| [part-09.md](part-09.md) | 242 | selected benchmark rows |
| [part-10.md](part-10.md) | 157 | current open builder gap |
