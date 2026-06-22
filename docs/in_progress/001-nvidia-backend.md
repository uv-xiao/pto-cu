# NVIDIA Backend Ultimate Goal

## Purpose

This umbrella goal tracks the long-running NVIDIA backend restart for PTO and
`simpler`. The goal is to move from source exploration through generated CUDA
kernels, persistent-device execution, multi-GPU experiments, and serving
evidence without relying on private terminal scrollback or one oversized dirty
branch.

The final target remains: prove a CUDA/NVIDIA flow that can run
simpler-programmed kernels on H200-class hardware and connect those kernels to
a serving path for DeepSeek-V4-Flash on 2 H200 GPUs with the requested
long-context target.

## Current Operating State

The original restart instruction exists as dispatcher context, not as a
tracked root file on `main`. The current repository is a normal GitHub repo
again: restart work is split into PR-sized branches and durable tracking lives
under `docs/in_progress/`.

The dispatcher session owns planning, review, merge decisions, and tracking
docs. Feature implementation should happen in child Codex sessions or bounded
worker branches, with each accepted slice represented by a GitHub PR.

## Non-Goals

- Do not preserve prior CUDA artifacts only because they existed in the
  historical restart workspace.
- Do not wrap PTO-ISA as the NVIDIA compiler path. Triton/Gluon serves the
  generated GPU-code role directly for this goal.
- Do not claim production readiness from smoke tests, local-only runs, or
  synthetic microbenchmarks.
- Do not append historical captures into `.agents/` when a shorter reusable
  rule, skill, or policy can carry the lesson.

## Source And Evidence Discipline

- External source checkouts, paper PDFs, extracted text, and exploratory notes
  belong under gitignored temporary directories.
- Primary external sources must be recorded with exact URL, retrieval method,
  local artifact location, and the reason the source matters.
- Claims about Triton/Gluon, Mirage, vLLM, FlashInfer, Ray, NCCL, UCCL, or
  DeepSeek serving must cite current source evidence.
- Hardware claims require exact commands, machine class, GPU model, commit,
  runtime configuration, and artifact location.
- A100/H200 paired claims require paired evidence. Local-only evidence must be
  described as local-only.
- Public contracts must have tests and docs in the same child slice.

## Scope And Artifact Map

- `.agents/`: Codex operating rules, skills, worker roles, and reusable
  policies for this long-running goal.
- `docs/in_progress/`: umbrella goal, dispatch log, worker prompts,
  preparation notes, source manifests, and active design sketches.
- `tmp/`: cloned repositories, papers, source notes, and raw local or remote
  artifacts that are not committed.
- `src/cuda/`: NVIDIA platform and runtime implementation.
- `simpler_setup/`: build, compiler, platform discovery, and CUDA/Gluon
  integration support.
- `python/` and pypto-facing surfaces: user-visible runtime and target
  integration APIs.
- `examples/cuda/` and `tests/`: executable slices for generated kernels,
  persistent-device megakernels, distributed behavior, and serving adapters.
- `docs/nvidia-backend/`: review-facing architecture, implementation, feature,
  evaluation, and changelog material promoted from `docs/in_progress/`.

## Required Outcomes

- A selected Codex skill set is installed or represented in `.agents/` for
  this branch, with source-platform assumptions translated to Codex and
  tmux-supervised Codex sessions.
- External sources from the restart instruction are cloned or downloaded under
  `tmp/` and indexed in source inventory docs.
- The CUDA evaluation skill stays operational and concise rather than becoming
  a historical capture dump.
- A Gluon generator/adapter can generate representative GEMM and
  flash-attention GPU kernels that run correctly on NVIDIA GPUs with
  performance evidence.
- The simpler NVIDIA platform supports a path where the Ascend
  host/AICPU/AICore model maps to a persistent-device megakernel containing
  orchestrator, scheduler, and worker behavior inside CUDA execution.
- pypto can target the simpler NVIDIA path, with extension points for features
  such as event tensors when justified by the design.
- The distributed compiler/runtime direction is evaluated against Ray, NCCL,
  UCCL, and related projects, with a documented path for fusing communication
  into large runtime kernels.
- A multi-GPU MoE dispatch/combine experiment runs through simpler NVIDIA
  mechanisms, aligned with the cited MoE dispatch/combine paper.
- A serving path through pto-serving or vLLM launches simpler NVIDIA kernels.
- DeepSeek-V4-Flash serving on 2 H200 GPUs with the requested long-context
  target produces correct output texts and records the full evidence chain.
- Each reviewable slice is tracked by an appropriate GitHub PR with focused
  scope, verification, and detailed description.

## Goal Mode And PR Slicing

This file is an ultimate-goal umbrella note, not a single-PR contract. Work
proceeds through dispatcher-managed child PRs.

- The dispatcher owns this umbrella note, child-PR sequencing, dispatch log,
  progress reports, shared contracts, and promotion into stable docs.
- Each child PR owns one coherent slice with its own branch, PR description,
  local verification, and merge decision.
- Workers may own one child PR but must not dispatch nested workers.
- Dependency PRs are required before adding reusable framework behavior or
  shared machinery that is not local to one child slice.
- Every dispatch, branch, PR, merge decision, scope change, and handoff is
  recorded in `docs/in_progress/nvidia_backend/dispatch_log.md`.

## Dispatcher Resume Contract

When a dispatcher starts or resumes this goal:

1. Read this file, the dispatch log, source inventory, current slicing plan,
   and task-relevant `.agents/` guidance.
2. Audit current repository state against the required outcomes above.
3. Record the audit summary and next child slice in the dispatch log.
4. Choose one reviewable child PR or dependency PR.
5. If launching a worker, record the exact objective, branch, allowed files,
   expected PR slot, and verification commands before starting Codex.

## Acceptance Criteria

- Source acquisition from the restart instruction is complete under `tmp/` and
  recorded in a source inventory or manifest.
- The selected Codex skill stack is present, minimal, and documented for Codex
  supervising Codex sessions.
- NVIDIA platform and runtime contracts are implemented with tests and docs.
- Gluon-generated GEMM and flash-attention kernels run correctly and have
  performance evidence.
- Simpler-programmed fused attention or MoE megakernels run successfully on
  real H200 hardware.
- A multi-GPU MoE dispatch/combine path runs with documented distributed
  communication decisions.
- Serving launches simpler NVIDIA kernels and DeepSeek-V4-Flash produces
  correct output texts on 2 H200 GPUs at the requested context target.
- GitHub PRs, dispatch logs, docs, tests, and hardware artifacts prove every
  accepted slice.

## Open Review Questions

- Which serving path should become the first integration target after source
  audit: pto-serving or vLLM?
- Which distributed communication layer should be the first dependency
  experiment: NCCL, UCCL, Ray orchestration, or a smaller direct CUDA proof?
- Which historical CUDA artifacts still deserve preservation as implementation
  seeds, and which should be retired as prior-branch residue?
