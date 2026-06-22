# Restart PR Slicing Plan

This plan tracks the current PR-sized route for continuing the NVIDIA backend
restart. The dirty orphan workspace is historical input only; new work starts
from `main` and lands through focused GitHub PRs.

## Current Baseline

- Base branch: `main`.
- Current accepted `main`: `902804ff`, after PR #145
  (`Define UCCL EP runtime fusion contract`).
- Repository hygiene PRs have already moved agent guidance to `.agents/`,
  added interval-based Codex goal monitoring, and merged the latest
  FlashAttention append coverage slice.
- PR #143 added an explicit `--with-uccl-ep-fused-boundary` status gate. It
  records `status: unsupported` after the UCCL-EP handoff passes because
  `persistent_device_uccl_ep_runtime_fusion` is missing. It is not fused
  cross-GPU expert-parallel MoE evidence.
- PR #145 accepted the design/dependency contract for
  `persistent_device_uccl_ep_runtime_fusion`. It is not an implementation and
  does not change the PR #143 evidence status.
- The abandoned branch `nvidia-uccl-ep-runtime-fusion-impl-h200` attempted an
  implementation after PR #145 but was rejected before push or PR because it
  synthesized pass evidence from handoff metadata instead of implementing real
  runtime-fusion ownership.
- The current handoff branch `nvidia-uccl-fusion-impl-blocked-handoff` owns
  only this slicing note, the dispatch log update, and the goal rollup update.

## Rules

- Each PR must stage only its owned paths. It must not bundle unrelated dirty
  worktree state.
- Use a fresh branch from current `main` for every child slice.
- Keep `docs/in_progress/nvidia_backend/dispatch_log.md` current for every
  branch, verification command, PR, merge decision, and handoff.
- Use interval-based monitor summaries before opening full tmux or transcript
  captures.
- Do not promote `docs/in_progress/` into `docs/nvidia-backend/` until the
  corresponding PR has been reviewed and accepted.

## Completed Bootstrap Slices

- PR #137: Codex goal monitor recurring tick support.
  - Result: merged as `7ab6dad31a6641c160305299a72467c74689486a`.
- PR #138: consolidate agent guidance under `.agents`.
  - Result: merged as `9e793ab1f51ccf2ffbf5120005acc2cddf42b843`.
- PR #136: broader Gluon FlashAttention append sweep coverage.
  - Result: merged as `c3964aad9204d0fbd0042ebbd7f88309530b80d2`.
- PR #139: restart tracking restoration.
  - Result: merged as `3722ad7efd7257fcc3807111aa449bfb49c57ea3`.
- PR #142: NVIDIA backend goal status rollup.
  - Result: merged as `afe82fb78783f24f36c01f41d9086064832cfaee`.
- PR #143: UCCL-EP fused boundary status.
  - Result: merged as `f73620c613b7a97c352384d6e90f32ae8c4106cd`.
  - Result type: structured unsupported boundary, not fused execution
    evidence.
- PR #144: refresh NVIDIA fused boundary status.
  - Result: merged as `f021845a94523664c2042ea7d8fd0dfb8a08d6cb`.
- PR #145: UCCL-EP runtime fusion contract.
  - Result: merged as `902804ff0bc9430448323240a77ebd1e12d775e8`.
  - Result type: design/dependency contract only, not fused execution
    evidence.

## Restored Tracking Surface

The restored tracking surface now includes a concise umbrella goal,
source/skill notes, artifact audit, slicing plan, and dispatch log so future
workers have durable context without committing a scrollback-sized historical
log.

PR: <https://github.com/uv-xiao/pto-cu/pull/139>.

Restored paths:

- `docs/in_progress/001-nvidia-backend.md`
- `docs/in_progress/nvidia_backend/source_manifest.md`
- `docs/in_progress/nvidia_backend/skill_selection.md`
- `docs/in_progress/nvidia_backend/cuda_eval_script_selection.md`
- `docs/in_progress/nvidia_backend/current_cuda_artifact_audit.md`
- `docs/in_progress/nvidia_backend/pr_slicing_plan.md`
- `docs/in_progress/nvidia_backend/dispatch_log.md`

Verification gate:

```bash
git diff --check
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

Non-claims:

- This PR does not implement new CUDA runtime behavior.
- This PR does not run new H200 kernels.
- This PR does not prove new serving or DeepSeek output behavior.

## Current Status Rollup

The current acceptance-area audit lives in
`docs/in_progress/nvidia_backend/goal_status_rollup.md`. It separates
DeepSeek/vLLM serving evidence from simpler-nv kernel integration evidence and
selects exactly one next worker slice. After PR #143, the UCCL-EP fused
boundary worker is complete as an unsupported-boundary status slice; it is no
longer a next candidate. After PR #145, the runtime-fusion design contract is
also complete as a design-only slice; it is not implementation evidence.

## Current Slice

This docs-only handoff slice records the abandoned implementation attempt:

- Branch: `nvidia-uccl-fusion-impl-blocked-handoff`.
- Scope: docs-only handoff for the rejected
  `nvidia-uccl-ep-runtime-fusion-impl-h200` attempt.
- Objective: record that local commit `8c7b3715` was abandoned because it
  synthesized `status: passed`, `actual_fused_cross_gpu_execution: true`,
  a payload ownership token, and a transition log from existing UCCL-EP
  handoff metadata. Synthetic pass evidence derived from handoff metadata is
  invalid.
- Non-claims: not DeepSeek serving, not vLLM plugin integration, not RDMA or
  multi-node evidence, not throughput or latency evidence, and not actual
  fused cross-GPU expert-parallel MoE execution.

## Next Dependency Slice

The next PR-sized dependency should not try to flip
`persistent_device_uccl_ep_runtime_fusion.status` to `passed`. It should first
add real payload provenance that is produced by the UCCL-EP adapter and the
persistent-device graph, rather than synthesizing ownership from handoff
metadata.

Recommended branch:
`nvidia-uccl-ep-adapter-payload-provenance`.

Objective: extend the UCCL-EP adapter handoff or fused-boundary result shape so
it records only real data emitted by the participating components:

- UCCL-EP adapter dispatch/combine descriptor provenance, including the
  adapter-reported token count, hidden size, top-k, expert count, dtype,
  metadata shapes, rank results, and UCCL capability id;
- persistent-device graph payload provenance, including graph descriptor id,
  device ids, rank/device mapping, source digest, and bridge digest;
- an explicit statement that no shared ownership token or lifetime transition
  log exists yet unless a runtime component actually creates and transfers it;
- `persistent_device_uccl_ep_runtime_fusion.status: unsupported` until a
  runtime-owned cross-component boundary exists.

This dependency may be an implementation slice if it adds real adapter-produced
data to the existing JSON, or a runtime/platform spike if the codebase first
needs to decide where cross-component ownership state can live behind
`ChipWorker` and the CUDA runtime boundary. It must not select a path whose
only route to `passed` is fabricated ownership or lifetime transitions.

A later implementation PR may reuse
`examples/cuda/persistent_moe_dispatch_combine.py
--with-uccl-ep-fused-boundary`, but it may claim actual fused cross-GPU
expert-parallel MoE execution only after a fresh H200 result proves real
payload ownership, rank/device mapping, boundary status fields, failure modes,
and `actual_fused_cross_gpu_execution: true`.

Serving promotion and in-progress doc retirement remain deferred until this
communication dependency boundary is reviewable or the dispatcher explicitly
chooses a different branch from current `main`.
