# CUDA Backend Status: Persistent-Device Runtime

## Persistent-Device Runtime

The CUDA `persistent_device` runtime is implemented as a set of tracer-bullet
execution modes:

- direct descriptor-array persistent executor;
- scheduler/worker bounded ready queue;
- bounded ring wraparound with capacity smaller than task count;
- generated-dispatch DAG with fan-in counters;
- five-task DAG-chain runtime graph descriptor;
- six-task scratch-reuse DAG descriptor;
- four-task multi-fan-in graph descriptor with three independent producers,
  scalar scale metadata, a third-tensor final join, and two scheduler blocks;
- nine-task layered-cross graph descriptor with three roots, cross-layer
  joins, a side branch, scalar metadata, and a stable third-tensor input;
- tensor-tile DAG descriptor with rows/cols/inner/stride metadata;
- tensor-core tile DAG descriptor with a block-wide CUDA WMMA
  `m16n16k8`/TF32/F32 generated-dispatch task body;
- scalar-argument DAG descriptors for single-tensor scale,
  mixed tensor/scalar AXPY-style, and two-scalar affine task bodies.
- third tensor-argument DAG descriptor for a generated-dispatch triad task
  body.
- fourth tensor-argument DAG descriptor for a generated-dispatch quad task
  body.
- generic tensor/scalar argument slots in the persistent DAG descriptor, with
  a generated-dispatch task reading `tensor_args[0]`, `tensor_args[1]`,
  `scalar_args[0]`, and `scalar_args[1]`.
- unary generated-dispatch DAG descriptor for a task body that reads one
  tensor input and leaves the second tensor pointer unused.
- device-side scheduler diagnostics for unsupported generated-dispatch
  `func_id` values, invalid dependent task IDs, out-of-range dependent
  spans, fan-in underflow, duplicate dependents, self-dependents, and
  initial-fan-in mismatch.
- device-side scheduler diagnostics for malformed graphs that have no
  zero-fan-in root, or that publish some ready roots but exhaust work before
  every task completes.
- explicit resource-policy smoke metadata for the current single scheduler
  block, configurable queue/DAG worker blocks, direct-mode worker blocks per
  task, and callable `stream_id`.
- prepared-callable repeat-run lifecycle metadata for direct, queue, and DAG
  modes, with queue counters/flags and DAG graph state reset between launches.

The persistent DAG path compiles generated CUDA source with `nvcc` and stores
the generated source, PTX, and manifest under
`build/cache/cuda/onboard/persistent_device/callables/`.
The smoke and benchmark path now reaches that artifact compiler through
`KernelCompiler(platform="cuda").compile_cuda_persistent_device(...)`, which
accepts task source files plus `func_id` metadata, lowers task-body style
sources through the same `CudaTaskBody` wrapper contract as `host_schedule`,
and composes the generated dispatch entry.
`SceneTestCase` L2 compilation accepts `CALLABLE["cuda"]` specs for
`persistent_device`, compiles task-body sources through the same
`KernelCompiler` entry point, registers the prepared raw callable through the
normal L2 `Worker`, builds `persistent_dag_fork_join_f32`,
`persistent_dag_chain_f32`, `persistent_dag_reuse_f32`, and
`persistent_dag_scalar_scale_f32`, `persistent_dag_scalar_axpy_f32`, and
`persistent_dag_scalar_affine_f32` scalar descriptors,
`persistent_dag_tensor_tile_f32` state objects,
`persistent_dag_triad_f32` third-tensor descriptors,
`persistent_dag_quad_f32` fourth-tensor descriptors,
`persistent_dag_generic_args_f32` generic tensor/scalar argument descriptors,
`persistent_dag_graph_f32` explicit and tensor-flow-inferred graph
descriptors, and
`persistent_dag_unary_square_f32` unary descriptors, and
`persistent_dag_tensor_core_tile_f32` WMMA tensor-core descriptors from normal
`TaskArgsBuilder` CPU tensors and scalars, and validates real copied-back CUDA
output data. The explicit graph descriptor adapter resolves per-task
`scalar0`, `scalar1`, and generic `scalar_args` entries from either numeric
literals or `TaskArgsBuilder` scalar names, so graph descriptors use the same
scalar argument flow as the fixed scalar descriptor adapters. The scene-test
persistent-device compiler path also forwards callable
`stream_id` into the prepared CUDA manifest, so these L2 tests can run on a
selected non-default runtime stream. After each persistent-device scene-test
launch, the L2 path now copies back device scheduler counters and raises on
nonzero scheduler errors or incomplete DAG execution, so scheduler failures
are visible even when a diagnostic test intentionally skips golden comparison.
The no-torch
persistent smoke path also validates a generated-dispatch triad descriptor
with a third tensor pointer field, a quad descriptor with third and fourth
tensor pointer fields, a generic-argument descriptor, and a generated-dispatch
unary-square descriptor with a single tensor input.
The `graph_descriptor_submits` no-torch smoke now routes through
`simpler_setup/cuda_normal_graph.py`, which lowers normal graph node keys and
`depends_on` edges into the persistent DAG fan-in array, flattened dependent
array, and per-task dependent spans before ABI materialization.
The host-schedule scene path also accepts the neutral
`elementwise_binary_f32` adapter for non-addition task bodies that still use
the current `(a, b, out, n)` launch ABI. It accepts `elementwise_unary_f32`
for unary `(a, out, n)` task bodies, `elementwise_scale_f32` for scalar
`(a, out, alpha, n)` task bodies, and `elementwise_axpy_f32` for mixed
tensor/scalar `(a, b, out, alpha, n)` task bodies. It also accepts
`elementwise_affine_f32` for two-scalar affine
`(a, b, out, alpha, beta, n)` task bodies and `elementwise_triad_f32` for
three-input `(a, b, c, out, n)` task bodies, `elementwise_quad_f32` for
four-input `(a, b, c, d, out, n)` task bodies, and
`elementwise_generic_args_f32` for the host-schedule generic tensor/scalar
argument slots. The original two-slot launch ABI remains available, and the
host runtime also accepts the four tensor/scalar slots already represented by
`CudaVectorGenericArgs`.
The no-torch Worker smoke can validate that same non-addition host-schedule
ABI with `--op mul`, unary ABI with `--op square`, scalar ABI with
`--op scale`, mixed tensor/scalar ABI with `--op axpy`, and two-scalar affine
ABI with `--op affine`, three-input ABI with `--op triad`, and four-input ABI
with `--op quad`, which keeps H200 coverage available when the remote Python
environment lacks `torch`.

Evidence:

- `tests/ut/py/test_cuda_backend.py` runs persistent-device smoke tests with
  real CUDA data when `nvcc` is available.
- `tests/ut/py/test_cuda_persistent_codegen.py` covers generated dispatch,
  device scheduler diagnostic fields, tensor descriptor fields, shared
  task-body wrapper generation, host-schedule and persistent-device manifest
  writing, and cache reuse.
- `tests/ut/py/test_cuda_kernel_compiler.py` covers both CUDA
  `KernelCompiler` entry points.
- `simpler_setup/cuda_normal_graph.py` provides the first small normal-graph
  lowering boundary used by a persistent-device smoke path.
- `simpler_setup/cuda_preflight.py` gives CUDA real-data tests one shared
  preflight path for `nvcc`, `nvidia-smi`, and driver visibility.
- `simpler_setup/cuda_callable_compiler.py` contains the generated-dispatch
  source renderer, shared task-body wrapper renderer, prepared-callable
  manifest helpers, and offline `nvcc` compile helper.
