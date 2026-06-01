# NVIDIA CUDA Backend Design

This page is the stable entrypoint for the CUDA backend design. The detailed
architecture notes are split into focused pages so reviewers can inspect one
contract at a time while preserving the original `overall.md` path used by the
README and project preparation guide.

The CUDA backend goal is to fit NVIDIA GPUs into the existing `simpler` / PTO
runtime architecture without leaking CUDA-specific details into the Python user
API. For adjacent details, see [flows.md](flows.md),
[persistent-device.md](persistent-device.md), [evaluation.md](evaluation.md),
and [status.md](status.md).

## Design Map

- [Architecture and scope](overall/architecture-and-scope.md): current PTO
  runtime shape, CUDA constraints, naming, and phase scope.
- [Runtime shape](overall/runtime-shape.md): host-scheduled and persistent
  device runtime responsibilities.
- [Build and kernel contract](overall/build-and-kernel-contract.md): directory
  layout, role-keyed binaries, toolchains, and CUDA callable compilation.
- [Semantics, testing, and roadmap](overall/semantics-testing-roadmap.md):
  runtime concept mapping, test strategy, open decisions, and first slice.
- [Sources](overall/sources.md): external CUDA references used by the design.

## Current Position

The current branch already implements CUDA role-keyed runtime binaries,
`host_schedule` and `persistent_device` scaffolding, CUDA example artifacts,
benchmark-viewer data, and review guards. The focused status pages remain the
source of truth for what is implemented, what was verified, and what still
blocks paper-ready claims.
