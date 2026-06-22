# Restart PR Slicing Plan

This plan tracks the current PR-sized route for continuing the NVIDIA backend
restart. The dirty orphan workspace is historical input only; new work starts
from `main` and lands through focused GitHub PRs.

## Current Baseline

- Base branch: `main`.
- Current accepted `main`: `2e9b01450efb709ed4e42f80a5128a01e8f9ad21`,
  after PR #148 (`Refresh NVIDIA status after payload provenance`).
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
- PR #146 recorded the abandoned
  `nvidia-uccl-ep-runtime-fusion-impl-h200` attempt as invalid because it
  synthesized pass evidence from handoff metadata instead of emitting real
  runtime-fusion ownership evidence.
- PR #147 recorded UCCL-EP adapter descriptor/rank payload provenance and
  persistent-device graph payload provenance. The H200 fused-boundary command
  exited `unsupported` as expected,
  `persistent_device_uccl_ep_runtime_fusion.status` remains `unsupported`,
  `actual_fused_cross_gpu_execution` remains `false`, and no shared payload
  ownership token or lifetime transition log exists. It is accepted
  provenance-only evidence, not fused execution evidence.
- PR #148 recorded the post-PR #147 status refresh and selected the
  `nvidia-uccl-ep-runtime-fusion-readiness` dependency slice. It did not
  change runtime behavior, result shape, or fused-execution evidence status.
- PR #149 defined the implementation-readiness map for the runtime-owned
  descriptor boundary and selected
  `nvidia-uccl-ep-runtime-fusion-impl-runtime-owned-descriptor` as the next
  narrow implementation attempt.
- The current implementation branch found no real runtime-owned
  `persistent_device_uccl_ep_runtime_fusion` coordinator behind the CUDA
  runtime / `ChipWorker` boundary. It therefore keeps the fused-boundary
  result `unsupported` and adds local guards that reject fabricated or
  incomplete pass evidence.
- The abandoned branch `nvidia-uccl-ep-runtime-fusion-impl-h200` attempted an
  implementation after PR #145 but was rejected before push or PR because it
  synthesized pass evidence from handoff metadata instead of implementing real
  runtime-fusion ownership.

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
- PR #146: abandoned UCCL-EP fusion attempt handoff.
  - Result: merged as `e0ee287939f7798578e318bf16017fe4939371c2`.
  - Result type: invalid implementation attempt record only, not fused
    execution evidence.
- PR #147: UCCL-EP adapter payload provenance.
  - Result: merged as `6405dfbd8b403b8d6a0e82813e185c209d4d7e08`.
  - Result type: provenance-only unsupported-boundary evidence, not fused
    execution evidence.
- PR #148: refresh NVIDIA status after payload provenance.
  - Result: merged as `2e9b01450efb709ed4e42f80a5128a01e8f9ad21`.
  - Result type: status/slicing refresh only, not fused execution evidence.

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
After PR #147, payload provenance is accepted as provenance-only evidence; it
is no longer a next candidate and does not change fused-execution status.

## Accepted Payload Provenance Slice

This section records the dependency slice selected after PR #146 and accepted
by PR #147. It is not actual fused cross-GPU expert-parallel MoE execution.

The accepted PR-sized dependency did not try to flip
`persistent_device_uccl_ep_runtime_fusion.status` to `passed`. It recorded
real payload provenance produced by the UCCL-EP adapter and the
persistent-device graph, rather than synthesizing ownership from handoff
metadata.

Accepted branch:
`nvidia-uccl-ep-adapter-payload-provenance`.

Accepted objective: extend the UCCL-EP adapter handoff and fused-boundary
result shape so it records only real data emitted by the participating
components:

- UCCL-EP adapter dispatch/combine descriptor provenance, including the
  adapter-reported token count, hidden size, top-k, expert count, dtype,
  metadata shapes, rank results, and UCCL capability id;
- persistent-device graph payload provenance, including graph descriptor id,
  device ids, rank/device mapping, source digest, and bridge digest;
- an explicit statement that no shared ownership token or lifetime transition
  log exists yet unless a runtime component actually creates and transfers it;
- `persistent_device_uccl_ep_runtime_fusion.status: unsupported` until a
  runtime-owned cross-component boundary exists.

PR #147 met this objective as provenance-only evidence. The H200 evidence
exited with status `unsupported`, as expected. It records
`persistent_device_uccl_ep_runtime_fusion.status: unsupported`,
`actual_fused_cross_gpu_execution: false`, no shared payload ownership token,
and an empty lifetime transition log. It must not be cited as fused execution
evidence.

## Selected Next Dependency Slice

This section records the selected PR-sized slice from current `main` after
PR #148. The current slice is a conservative docs/design readiness slice, not
a runtime implementation attempt.

Recommended branch:
`nvidia-uccl-ep-runtime-fusion-readiness`.

Objective: define an implementation-readiness map for the real
`persistent_device_uccl_ep_runtime_fusion` boundary now that PR #145 defines
the contract and PR #147 records accepted provenance-only input fields. The
slice should answer where runtime-owned shared payload descriptors live, which
component records the ownership token and lifetime transition log, which
failure states are mandatory, and what local and H200 evidence will be
required before a later implementation may report `passed`.

Allowed future scope:

- `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`
- `docs/in_progress/nvidia_backend/communication_selection.md`
- `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`
- `docs/in_progress/nvidia_backend/pr_slicing_plan.md`
- `docs/in_progress/nvidia_backend/dispatch_log.md`
- `tests/ut/py/test_nvidia_review_artifacts.py`

Verification commands:

```bash
git diff --check
```

```bash
git diff --cached --check
```

```bash
npx --no-install markdownlint-cli2 \
  docs/in_progress/nvidia_backend/communication_runtime_boundary.md \
  docs/in_progress/nvidia_backend/communication_selection.md \
  docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md \
  docs/in_progress/nvidia_backend/pr_slicing_plan.md \
  docs/in_progress/nvidia_backend/dispatch_log.md
```

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python .agents/checks/check_nvidia_review_ready.py
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python -m pytest tests/ut/py/test_nvidia_review_artifacts.py -q
```

H200 evidence requirement: no fresh H200 command is required if the slice is
docs/design/test-guard only. If it changes example behavior or result shape,
it must run a fresh H200 fused-boundary command and still report
`unsupported` unless a real runtime-owned payload boundary exists.

A later implementation PR may reuse
`examples/cuda/persistent_moe_dispatch_combine.py
--with-uccl-ep-fused-boundary`, but it may claim actual fused cross-GPU
expert-parallel MoE execution only after a fresh H200 result proves real
payload ownership, rank/device mapping, boundary status fields, failure modes,
and `actual_fused_cross_gpu_execution: true`.

Serving promotion and in-progress doc retirement remain deferred until this
communication dependency boundary is reviewable or the dispatcher explicitly
chooses a different branch from current `main`.

## Current Guard-Only Implementation Handoff

The dispatcher selected exactly one branch after the readiness map:
`nvidia-uccl-ep-runtime-fusion-impl-runtime-owned-descriptor`.

The branch could not honestly implement a runtime-owned shared payload
descriptor because the lower-level
`persistent_device_uccl_ep_runtime_fusion` coordinator does not exist behind
the CUDA runtime / `ChipWorker` boundary. The compiled persistent-device
runtime role files remain placeholders, and the active CUDA platform runner
launches generated persistent DAG kernels without a UCCL-EP runtime
dispatch/combine ownership handoff.

Accepted scope for this blocked implementation handoff:

- keep the normal `--with-uccl-ep-fused-boundary` result `unsupported`;
- add local guard code that rejects fabricated or incomplete pass evidence;
- add unit coverage for missing ownership tokens, mismatched tokens, double
  release, use-after-release, leaked ownership, rank/device mismatch, and
  untrusted adapter/provenance pass fields;
- update the communication boundary docs and dispatch log with the blocked
  lower-level dependency.

No fresh H200 fused-boundary success result is required or claimed for this
blocked handoff because the implementation cannot truthfully emit real
runtime-owned descriptor evidence. A future implementation must first add the
runtime-owned fusion coordinator behind the CUDA runtime / `ChipWorker`
boundary before reporting `persistent_device_uccl_ep_runtime_fusion.status:
passed` or `actual_fused_cross_gpu_execution: true`.
