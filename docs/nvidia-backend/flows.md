# NVIDIA Backend Flow Details

This document expands the CUDA backend design with operational flow details:
how `simpler` compiles and launches today, where CUDA differs, which CUDA API
surface should own each step, and how lifecycle, memory, and callable concepts
map onto CUDA.

## Review Map

- [Scope](flows/scope.md) (14 lines)
- [User-Facing Compile and Launch Flow](flows/compile-launch-flow.md) (103 lines)
- [Runtime Flow Comparison](flows/runtime-flow-comparison.md) (142 lines)
- [CUDA Runtime API vs Driver API](flows/runtime-vs-driver-api.md) (40 lines)
- [TileLang JIT Flow](flows/tilelang-jit-flow.md) (61 lines)
- [Lifecycle, Memory, and Callable Mapping](flows/lifecycle-memory-callable.md) (73 lines)
- [Design Consequences](flows/design-consequences.md) (24 lines)
- [Sources](flows/sources.md) (27 lines)

## Flow Summary

The CUDA backend should preserve the `simpler` user flow: compile a callable,
register it with a `Worker`, initialize the runtime, run with `TaskArgs` and
`CallConfig`, then copy results back through the existing worker boundary.
CUDA changes the runtime payload and launch path, not the top-level usage
model.

`host_schedule` maps naturally to host-enqueued CUDA work on streams.
`persistent_device` maps to a host-launched persistent executor with
scheduler/worker roles inside one CUDA grid. The runtime should use explicit
CUDA context and module ownership, with the Driver API owning dynamic module
loading and the Runtime API used only where it simplifies safe operations.
