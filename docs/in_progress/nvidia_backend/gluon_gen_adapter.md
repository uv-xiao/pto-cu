# Gluon-Gen Adapter Boundary

This note records the first `gluon-gen` boundary for the NVIDIA backend
restart. It is a source-generation adapter plus scalar GEMM harness, not a
performance claim.

## Current Contract

`KernelCompiler(platform="cuda").generate_gluon_kernel(...)` writes a
Triton/Gluon Python source file and JSON manifest through
`simpler_setup/gluon_gen.py`.

The manifest marks the compiler role as `pto-isa-replacement`. That phrase is
intentional: the NVIDIA path uses Triton/Gluon directly as the GPU codegen
surface, not PTO-ISA and not a wrapper around PTO-ISA.

Supported kernel in this PR:

- `gemm_f32`

The generated source imports `triton.experimental.gluon`, declares a
`@gluon.jit` kernel, and writes artifact metadata under the selected output
directory. Review and H200 commands use `tmp/gluon-*` output directories so
generated Gluon source and JSON manifests remain uncommitted. Unit tests
verify generation and the CUDA-only `KernelCompiler` entry point without
requiring Triton installation or a GPU.

`examples/cuda/gluon_gemm_f32.py` is the first skip-safe correctness harness
for this boundary. It always generates the reviewable `gemm_f32` artifact and
then either runs a PyTorch reference comparison on CUDA, or reports structured
`skipped` JSON when PyTorch, Triton/Gluon, or CUDA are unavailable. On H200,
use `--require-cuda` so a skip is treated as a validation failure.

The current `gemm_f32` source is deliberately scalar: each Gluon program
computes one output element. That keeps the first execution proof on the
smallest stable Gluon surface. Tensor-core layouts, flash-attention, MoE,
distributed execution, serving integration, and performance work remain
separate milestones.

## H200 Evidence

`docs/in_progress/nvidia_backend/gluon_gemm_h200.md` records the H200 command
and scalar result for this PR.

## What This Proves

- The CUDA compiler facade has a named Gluon adapter entry point.
- The generated artifact records the source hash, architecture target, tile
  shape, and `pto-isa-replacement` role.
- The GEMM harness gives reviewers one stable command for moving from generated
  source to scalar correctness evidence.

## What It Does Not Prove

- A local skip does not compile the Gluon source.
- Local skip-safe tests do not prove GEMM correctness.
- The scalar GEMM path does not prove tensor-core codegen.
- It does not prove flash-attention, MoE, distributed execution, serving, or
  performance.

Those claims need later PRs with exact commands, artifacts, correctness
checks, and benchmark evidence.
