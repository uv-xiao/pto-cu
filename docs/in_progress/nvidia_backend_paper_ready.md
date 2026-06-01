# NVIDIA Backend Paper-Ready Goal

## Purpose

This ultimate goal makes the NVIDIA backend a standalone pto-cu project
outcome. The final repository should support a CUDA backend path that humans
can review as an implementation, an evaluation artifact, and a future paper
submission story.

The goal is larger than one PR. It needs dispatcher-managed child PRs for CUDA
runtime work, benchmark infrastructure, examples, documentation, evidence
guards, source notes, and paper-ready evaluation.

## Problem

The current CUDA work has reviewable design documents, starter examples, and
a human-reviewable benchmark viewer. The remaining gap is rigor: the repository must
make benchmark status easy to inspect, keep documents tied to explicit code
evidence, record every architectural and evaluation change in changelog
reports, and grow evaluation until it can support a paper-quality claim.

CUDA also differs from the A2/A3 and A5 targets because it has no AICPU. The
host-schedule runtime can use host-side CUDA async APIs, streams, and events to
launch kernels. The persistent-device runtime needs a compiled device-side
scheduler inside the CUDA binary so scheduler warps can dispatch device task
functions onto worker warps without rewriting every task as a separate
`__global__` kernel.

## Non-Goals

- Do not write, push, edit PRs, or change settings in upstream repositories.
- Do not treat A2/A3 or A5 CI as required for this standalone NVIDIA branch.
- Do not claim paper-ready performance before full baseline reproduction,
  artifact capture, and reviewer-readable analysis exist.
- Do not merge design text that describes behavior without code evidence or a
  clearly marked planned status.

## Source And Evidence Discipline

- Used external sources, cloned baselines, extracted paper text, and source
  notes stay under `tmp/` for local inspection.
- Stable design documents cite repo-relative code paths, scripts, viewer data,
  and changelog reports.
- Every implemented feature described by docs must have explicit
  `evidence_refs` or path-level evidence in the same child PR.
- If implementation differs from the plan, update the plan and add a changelog
  report that states the old assumption, the new behavior, and verification.
- Evaluation claims must name hardware, CUDA version, git commit, input shape,
  repeats, statistic, and baseline method.

## Scope And Artifact Map

- `src/cuda/`: CUDA runtime and platform implementation.
- `simpler_setup/` and `python/simpler/`: build, discovery, and user API
  integration for CUDA when needed.
- `examples/cuda/`: runnable NVIDIA examples matching evaluated workloads.
- `docs/nvidia-backend/`: stable design, runtime flow, evaluation, viewer,
  history, status, and changelog reports.
- `docs/in_progress/nvidia_backend_paper_ready*`: dispatcher-owned goal,
  work preparation, shared contracts, evaluation plan, and dispatch log.
- `.agents/`: working rules, skills, quality guards, and verification scripts.
- `tmp/`: local-only sources, cloned baselines, raw captures, reports, and
  paper notes.

## Goal Mode And PR Slicing

This file is an ultimate-goal umbrella note. Work proceeds through child PRs
that target `uv-xiao/pto-cu:main`.

- The dispatcher owns this umbrella note, child PR sequencing, dispatch log,
  progress reports, shared contracts, and final promotion into stable docs.
- Each child PR owns one reviewable slice with its own branch, verification,
  changelog report, and code-document evidence trail.
- Workers may own child PRs, but must not dispatch nested workers.
- Dependency PRs are required before shared schemas, tooling, or runtime
  abstractions are reused by multiple workers.
- Every dispatch, branch, PR, merge decision, scope change, and handoff is
  recorded through
  `docs/in_progress/nvidia_backend_paper_ready/dispatch_log.md`, with
  dated archive chunks linked from
  `docs/in_progress/nvidia_backend_paper_ready/dispatch_log/index.md`.

## Required Child Slices

- Current-state audit: compare branch status against this goal and record gaps.
- Viewer expansion: make benchmark setup, math/code explanations, run commands,
  method definitions, and results load from structured data.
- Evidence guard: enforce document-to-code evidence, readable abstractions,
  changelog coverage, examples, and source-note discipline.
- CUDA host-schedule maturity: validate launch, stream concurrency, memory
  lifecycle, callable ABI, and graph/capture comparison.
- CUDA persistent-device maturity: compile scheduler and task code together,
  map callable task functions to device dispatch, and document lifecycle.
- Examples: keep `examples/cuda/` aligned with evaluated workloads.
- Remote evaluation: support Git-based refresh when available and tree-sync
  fallback when remote Git credentials or network fail. This is the required
  remote evaluation fallback path.
- Paper-ready evaluation: reproduce or compare against MPK, VDCores, and the
  baselines used by their papers.

## Acceptance Criteria

- The benchmark viewer is human-reviewable and backed by versioned JSON data.
- NVIDIA examples match at least the smoke and benchmark workloads used in
  evaluation reports.
- Stable docs distinguish implemented behavior from planned behavior and link
  implemented claims to code evidence.
- Changelog reports state what code changed, how architecture quality changed,
  what evaluation ran, and what evidence proves it.
- Remote evaluation has both Git refresh and SSH tree-sync fallback paths.
- The paper evaluation plan covers MPK, VDCores, and their paper baselines.
- Final paper-grade results include correctness, latency, throughput,
  scheduler overhead, scaling, statistics, and raw artifacts for A100 and H200.
- The dispatcher log is complete enough for another Codex session or human
  reviewer to resume without private context.

## Review Questions

- Which CUDA runtime should be the first paper claim: host-schedule,
  persistent-device, or both as separate systems?
- Which MPK and VDCores workloads can be reproduced directly, and which should
  be represented by smaller controlled kernels in this repo?
- When should a planned design claim be promoted to implemented status?
- Which benchmarks need end-to-end LLM serving reproduction before a paper
  submission can be credible?
