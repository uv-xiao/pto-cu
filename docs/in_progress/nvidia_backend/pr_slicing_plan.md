# Restart PR Slicing Plan

This plan tracks the current PR-sized route for continuing the NVIDIA backend
restart. The dirty orphan workspace is historical input only; new work starts
from `main` and lands through focused GitHub PRs.

## Current Baseline

- Base branch: `main`.
- Current accepted `main`: `e33d232deccdf947b9c382a3605191d0d5ae0004`,
  after PR #168 (`Map UCCL EP validation policy`).
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
- PR #150 found no real runtime-owned
  `persistent_device_uccl_ep_runtime_fusion` coordinator behind the CUDA
  runtime / `ChipWorker` boundary. It therefore keeps the fused-boundary
  result `unsupported` and adds local guards that reject fabricated or
  incomplete pass evidence. It is accepted only as a guard-only blocked
  implementation handoff, not fused execution evidence.
- PR #151 recorded the post-PR150 status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-coordinator-boundary-map` as the next
  dependency slice. It did not change runtime behavior, result shape, or
  fused-execution evidence status.
- PR #152 mapped the runtime-owned coordinator boundary below the CUDA
  runtime / `ChipWorker` edge and selected
  `nvidia-uccl-ep-runtime-fusion-coordinator-entry-contract` as the next
  dependency slice. It did not implement runtime behavior, expand public
  `TaskArgs` / `CallConfig`, or change fused-execution evidence status.
- PR #153 defined the private
  `persistent_device_uccl_ep_runtime_fusion_entry` contract below
  `ChipWorker::run` and selected
  `nvidia-uccl-ep-runtime-fusion-private-entry-unsupported` as the next
  narrow implementation slice. It did not implement runtime behavior, expand
  a UCCL host-runtime ABI, or change fused-execution evidence status.
- PR #154 recorded the post-PR153 status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-private-entry-unsupported` as the next
  narrow implementation slice. It did not change CUDA runtime behavior,
  result shape, or fused-execution evidence status.
- PR #155 added the private CUDA host-side
  `persistent_device_uccl_ep_runtime_fusion_entry` request/result scaffold.
  The CUDA persistent DAG host-runtime path now records an unsupported
  runtime-fusion result from private state only. It did not report
  `persistent_device_uccl_ep_runtime_fusion.status: passed`, set
  `actual_fused_cross_gpu_execution: true`, expand public `TaskArgs` or
  `CallConfig`, expand a UCCL host-runtime ABI, or claim fresh H200 fused
  success.
- PR #156 recorded the post-PR155 status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request` as the next
  dependency slice. It did not change CUDA runtime behavior, result shape, or
  fused-execution evidence status.
- PR #157 attempted that request boundary but was closed invalid. It recorded
  the persistent DAG run `args` pointer, whose static type at the CUDA
  host-runtime call site is `PtoCudaPersistentDagArgs *`, as
  `PtoCudaRuntimeFusionRequest::chip_storage_task_args` and labeled it
  `sizeof(ChipStorageTaskArgs)`. That was not a real
  `ChipStorageTaskArgs *` from `ChipWorker::run`, so it is a blocked handoff,
  not accepted implementation evidence.
- PR #158 fixed the Codex monitor transcript lookup and is already on `main`.
  It is unrelated to the NVIDIA runtime-fusion request boundary.
- PR #159 recorded PR #157 as a closed invalid ChipStorage request handoff,
  kept the CUDA runtime-fusion code state unclaimed, and selected the private
  request-envelope dependency that PR #160 later implemented.
- PR #160 added the private CUDA run envelope and host-runtime hook while
  keeping the fused-boundary result unsupported. It is accepted only as a
  private request-envelope / host-runtime handoff dependency, not
  runtime-fusion success. It does not report
  `persistent_device_uccl_ep_runtime_fusion.status: passed`, set
  `actual_fused_cross_gpu_execution: true`, expand public `TaskArgs` or
  `CallConfig`, expand a UCCL host-runtime ABI, or claim fresh H200 fused
  success.
- PR #161 recorded the post-PR160 status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-runtime-args-handoff-map` as the next
  conservative dependency slice. It did not change CUDA runtime behavior,
  result shape, or fused-execution evidence status.
- PR #162 accepted that runtime-args handoff map as a docs/test dependency
  slice only. It did not change CUDA runtime behavior, result shape, or
  fused-execution evidence status. The selected next slice is the narrow
  private host-runtime handoff implementation, not another handoff-map slice.
- PR #163 recorded the post-PR162 status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff` as the next
  narrow implementation slice. It did not change CUDA runtime behavior,
  result shape, or fused-execution evidence status.
- PR #164 added the private CUDA persistent DAG host-runtime handoff that
  associates real same-invocation `ChipStorageTaskArgs *` and
  `PtoCudaPersistentDagArgs *` pointers before the private runtime-fusion
  entry is requested. It is accepted only as a private handoff
  implementation, not as a coordinator, descriptor allocator, UCCL-EP runtime
  path, validation policy, capability metadata, pass evidence, fresh H200
  fused-success evidence, or fused-execution status change.
- PR #165 recorded the post-PR164 docs/test status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-capability-metadata-map` as the next
  dependency slice. It did not change CUDA runtime behavior, result shape, or
  fused-execution evidence status.
- PR #166 accepted only a private UCCL-EP capability metadata dependency map:
  capability id, world size, rank-to-device map, descriptor vocabulary,
  transport mode, adapter provenance handles, and setup/validation failure
  ownership. It did not implement a runtime-fusion coordinator, descriptor
  allocator, UCCL-EP runtime path, validation policy, CUDA runtime behavior,
  pass evidence, or H200 fused-success evidence.
- PR #167 recorded the post-PR166 status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-validation-policy-map` as the next
  dependency slice. It did not change CUDA runtime behavior, result shape, or
  fused-execution evidence status.
- PR #168 accepted only the private validation policy dependency map. It did
  not implement CUDA runtime behavior, descriptor allocation policy, UCCL-EP
  runtime dispatch, a coordinator, pass evidence, or H200 fused-success
  evidence.
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
- PR #149: UCCL-EP runtime fusion readiness.
  - Result: merged as `d7d1679d84ef08202e3a61a821613e031edd49bd`.
  - Result type: implementation-readiness map only, not fused execution
    evidence.
- PR #150: guard UCCL-EP runtime fusion evidence.
  - Result: merged as `a6378bfbf55b15be01c334f43332ccd20c160cfa`.
  - Result type: guard-only blocked implementation handoff, not fused
    execution evidence.
- PR #151: refresh NVIDIA status after runtime fusion guard.
  - Result: merged as `3548a5761c2785bc855d68ec53469651d2227096`.
  - Result type: status/slicing refresh only, not fused execution evidence.
- PR #152: UCCL-EP runtime fusion coordinator boundary map.
  - Result: merged as `8b5e8075000a2a3e35c4e71c5cb698224b003b44`.
  - Result type: coordinator-boundary map only, not fused execution evidence.
- PR #153: UCCL-EP coordinator entry contract.
  - Result: merged as `b58598490d37065e6c972eaaea6d4bc4900469c7`.
  - Result type: private entry-contract only, not fused execution evidence.
- PR #154: refresh NVIDIA status after entry contract.
  - Result: merged as `29da72a171b25deeeb53db399f9cdf54d38c647a`.
  - Result type: status/slicing refresh only, not fused execution evidence.
- PR #155: private UCCL-EP runtime fusion entry scaffold.
  - Result: merged as `d04732e3a5513d8172b41d0812f2d84065039526`.
  - Result type: private unsupported runtime scaffold only, not fused
    execution evidence.
- PR #156: refresh NVIDIA status after private entry scaffold.
  - Result: merged as `6b6b3f3756da2b3857c7206cb6625383a6dc0bd7`.
  - Result type: status/slicing refresh only, not fused execution evidence.
- PR #158: fix Codex monitor transcript lookup.
  - Result: merged as `41a9e1e4135313a9787386fb32c21f8b85254d4b`.
  - Result type: monitor tooling fix only, unrelated to CUDA runtime-fusion
    evidence.
- PR #159: record invalid ChipStorage request handoff.
  - Result: merged as `f1b4abb9c9544a71af70decc15bf1424837e0966`.
  - Result type: closed-invalid handoff record only, not fused execution
    evidence.
- PR #160: private CUDA runtime fusion request envelope.
  - Result: merged as `142132a2df296ce64e4cd2c17af909d619bcad22`.
  - Result type: private request-envelope and host-runtime handoff dependency
    only, not runtime-fusion success or fused execution evidence.
- PR #161: refresh NVIDIA status after PR 160.
  - Result: merged as `6026ed7cbfa1d4724e22e109bbd75c06d0e9f9a7`.
  - Result type: status/slicing refresh only, not runtime behavior or fused
    execution evidence.
- PR #162: map CUDA runtime args handoff.
  - Result: merged as `0ba8f30696132c06a3cd49b95fbd7bb46b8b9a99`.
  - Result type: docs/test dependency map only, not runtime behavior or fused
    execution evidence.
- PR #163: refresh NVIDIA status after runtime args map.
  - Result: merged as `cc26283be5b3355af8148a8e4ca5421d57c2ff80`.
  - Result type: status/slicing refresh only, not runtime behavior or fused
    execution evidence.
- PR #164: CUDA private runtime handoff.
  - Result: merged as `be914b97898468033c7f834dde0c43466353ac95`.
  - Result type: private host-runtime handoff only, not coordinator,
    descriptor allocator, UCCL-EP runtime path, validation policy, capability
    metadata, pass evidence, or fused execution evidence.
- PR #165: refresh NVIDIA status after PR 164.
  - Result: merged as `bb526ff6c3c21597cffe1acd34bf08158a947cc3`.
  - Result type: status/slicing refresh only, not runtime behavior or fused
    execution evidence.
- PR #166: map UCCL EP capability metadata.
  - Result: merged as `42b996666e279024b43f490a310c490a591a897d`.
  - Result type: private capability metadata dependency map only, not runtime
    behavior, validation policy, descriptor allocation, runtime-fusion
    coordinator behavior, pass evidence, or fused execution evidence.
- PR #167: refresh NVIDIA status after capability metadata.
  - Result: merged as `20b3e625ea8c9d6e4f06bb3992779b807f65acf9`.
  - Result type: status/slicing refresh only, not runtime behavior or fused
    execution evidence.
- PR #168: map UCCL EP validation policy.
  - Result: merged as `e33d232deccdf947b9c382a3605191d0d5ae0004`.
  - Result type: private validation policy dependency map only, not runtime
    behavior, descriptor allocation, UCCL-EP runtime dispatch, coordinator
    behavior, pass evidence, or fused execution evidence.

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
After PR #150, guard-only runtime-fusion evidence checks are accepted as a
blocked implementation handoff; they do not change fused-execution status.
After PR #151, the post-PR150 status refresh is complete; it is not a runtime
behavior or result-shape change. After PR #152, the coordinator-boundary map
is complete as a dependency slice; it is not a runtime behavior,
result-shape, or fused-execution evidence change. After PR #153, the
coordinator entry contract is complete as a private dependency slice; it is
not a runtime behavior, UCCL host-runtime ABI, result-shape, or
fused-execution evidence change. After PR #154, the post-PR153 status refresh
is complete. After PR #155, the private unsupported entry scaffold is
accepted as a request/result scaffold only; it is not fused execution
evidence. After PR #156, the post-private-entry status refresh is complete.
PR #157 remains closed invalid because it confused the persistent DAG args
pointer with a real `ChipStorageTaskArgs *`. PR #158 is a monitor tooling fix
only. PR #159 records the invalid PR #157 handoff and selected the private
request-envelope dependency. After PR #160, the private envelope and
host-runtime hook are accepted only as a dependency that prevents the
persistent DAG args pointer from being mislabeled as `ChipStorageTaskArgs *`;
it is not runtime-fusion success. After PR #161, the post-private-envelope
status refresh is complete. After PR #162, the runtime-args handoff map is
complete as a docs/test dependency slice; it is not a runtime behavior,
result-shape, or fused-execution evidence change. After PR #163, the
post-runtime-args status refresh is complete. After PR #164, the private
host-runtime handoff is accepted as a narrow implementation slice only; it
does not add coordinator-owned UCCL-EP runtime fusion evidence. After
PR #165, the post-PR164 status refresh is complete. After PR #166, the
private UCCL-EP capability metadata map is accepted as a dependency slice
only; it does not implement validation policy or runtime behavior. After
PR #167, the post-capability-metadata status refresh is complete. After
PR #168, the private validation policy map is accepted as a dependency slice
only; it does not implement descriptor allocation policy or runtime behavior.
The current accepted baseline is
`e33d232deccdf947b9c382a3605191d0d5ae0004`, and the next slice is exactly
one conservative dependency slice:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map`.

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

## Accepted Runtime Fusion Readiness Slice

This section records the dependency slice selected after PR #148 and accepted
by PR #149. It is not actual fused cross-GPU expert-parallel MoE execution.

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-readiness`.

Accepted objective: define an implementation-readiness map for the real
`persistent_device_uccl_ep_runtime_fusion` boundary now that PR #145 defines
the contract and PR #147 records accepted provenance-only input fields. The
slice should answer where runtime-owned shared payload descriptors live, which
component records the ownership token and lifetime transition log, which
failure states are mandatory, and what local and H200 evidence will be
required before a later implementation may report `passed`.

Accepted scope:

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

H200 evidence requirement: no fresh H200 command was required because the
slice was docs/design/test-guard only. A later branch that changes example
behavior or result shape must run a fresh H200 fused-boundary command and
still report `unsupported` unless a real runtime-owned payload boundary
exists.

A later implementation PR may reuse
`examples/cuda/persistent_moe_dispatch_combine.py
--with-uccl-ep-fused-boundary`, but it may claim actual fused cross-GPU
expert-parallel MoE execution only after a fresh H200 result proves real
payload ownership, rank/device mapping, boundary status fields, failure modes,
and `actual_fused_cross_gpu_execution: true`.

Serving promotion and in-progress doc retirement remain deferred until this
communication dependency boundary is reviewable or the dispatcher explicitly
chooses a different branch from current `main`.

## Accepted Guard-Only Implementation Handoff

The dispatcher selected exactly one branch after the readiness map, and
PR #150 merged it as
`a6378bfbf55b15be01c334f43332ccd20c160cfa`:
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

## Accepted Post-PR150 Status Refresh

PR #151 merged as `3548a5761c2785bc855d68ec53469651d2227096`:
`nvidia-goal-status-post-runtime-fusion-guard`
(`Refresh NVIDIA status after runtime fusion guard`).

The branch refreshed the goal status, slicing plan, dispatch log, and focused
review-artifact assertions after PR #150. It preserved the accepted evidence
boundary:

- PR #147 remains provenance-only unsupported-boundary evidence;
- PR #150 remains guard-only blocked implementation evidence;
- PR #151 is a status/slicing refresh only;
- no PR accepted `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no PR accepted `actual_fused_cross_gpu_execution: true`.

PR #151 selected
`nvidia-uccl-ep-runtime-fusion-coordinator-boundary-map` as the next
PR-sized dependency slice.

## Accepted Coordinator Boundary Map Slice

This section records the PR-sized slice selected after PR #151 and accepted by
PR #152. The branch made the lower-level coordinator boundary reviewable
because PR #150 confirmed that no runtime-owned coordinator exists below the
CUDA runtime / `ChipWorker` boundary and PR #151 only refreshed status.

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-coordinator-boundary-map`.

Objective: define the missing runtime-owned coordinator boundary for
`persistent_device_uccl_ep_runtime_fusion` before changing runtime behavior.
The slice mapped the coordinator's runtime owner, `ChipWorker` entry
point, descriptor allocation site, ownership token issuer, lifetime transition
state machine, failure-field responsibilities, local tests, and later H200
command evidence. It preserved the non-claims from PR #150: no actual
fused cross-GPU expert-parallel MoE execution, no fresh H200 fused-success
evidence, no accepted `persistent_device_uccl_ep_runtime_fusion.status:
passed`, and no accepted `actual_fused_cross_gpu_execution: true`.

Accepted scope:

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

H200 evidence result: no fresh H200 command was required because the slice was
docs/design/test-guard only and did not change example behavior or result
shape.

Coordinator-boundary requirements for this branch:

- runtime owner:
  `persistent_device_uccl_ep_runtime_fusion` inside the CUDA
  persistent-device runtime run context;
- reviewable entry point:
  `WorkerThread` chip dispatch to `ChipWorker::run`, then the CUDA
  host-runtime callable entry and persistent-device run context;
- descriptor allocation site:
  the CUDA persistent-device runtime run context, not example-side handoff
  metadata;
- ownership token issuer:
  the coordinator, with one token shared by dispatch and combine descriptors;
- lifetime state machine:
  `allocated`, `dispatch_ready`, `dispatch_in_flight`, `combine_ready`,
  `combine_in_flight`, `complete`, and `released`;
- failure-field responsibilities:
  setup, unsupported boundary, descriptor, rank/device, payload lifetime,
  transport, scheduler, validation, and fabricated or untrusted pass evidence;
- evidence boundary:
  PR #147 remains provenance-only unsupported-boundary evidence, PR #150
  remains guard-only blocked implementation evidence, and PR #151 remains a
  post-PR150 status refresh.

## Accepted Coordinator Entry Contract Slice

This section records the PR-sized slice selected after PR #152 and accepted
by PR #153. The branch made the private `ChipWorker::run` to CUDA
persistent-device host-callable entry contract reviewable before any
implementation constructs a coordinator.

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-coordinator-entry-contract`.

Objective: define the narrow private runtime entry contract that an
implementation branch needs before constructing a coordinator. This remained
a dependency slice because the coordinator-boundary map did not create
runtime code, descriptor memory, UCCL-EP runtime dispatch, or H200 pass
evidence. The accepted contract defines how `ChipWorker` and the CUDA host
runtime request, return, and validate coordinator result fields without
adding public `TaskArgs`, public `CallConfig`, or UCCL host-runtime ABI
fields.

Required contract content:

- private entry owner and name:
  CUDA persistent-device runtime owns
  `persistent_device_uccl_ep_runtime_fusion_entry`;
- request path:
  `ChipWorker::run` builds `ChipStorageTaskArgs` and requests coordinator
  construction through private CUDA host-runtime state, not public API fields;
- request fields:
  callable id, chip-local rank/device map, persistent graph descriptor handle,
  UCCL-EP capability metadata, descriptor allocation policy, validation
  policy, and output sink;
- result fields:
  coordinator status, descriptor allocation provenance, ownership token, state
  transitions, rank/device map, validation summary, and failure fields;
- forbidden evidence paths:
  example-side JSON, adapter-only provenance, handoff metadata, public
  `TaskArgs`, and public `CallConfig`;
- failure behavior:
  unsupported, setup, descriptor, rank/device, payload lifetime, transport,
  scheduler, validation, and fabricated or untrusted pass evidence stay
  explicit.

Allowed scope:

- `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`
- `docs/in_progress/nvidia_backend/communication_selection.md`
- `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`
- `docs/in_progress/nvidia_backend/pr_slicing_plan.md`
- `docs/in_progress/nvidia_backend/dispatch_log.md`
- focused review-artifact tests if the assertions need to pin the new
  contract

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

Required non-claims:

- no CUDA runtime behavior change;
- no UCCL host-runtime ABI expansion;
- no fresh H200 fused-success claim;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim.

PR #153 met this objective as a private entry-contract slice only. It did not
implement CUDA runtime behavior, change the fused-boundary example result
shape, create descriptor memory, emit a coordinator-owned ownership token,
claim fresh H200 fused success, or change the accepted unsupported evidence
status.

## Accepted Private Entry Unsupported Scaffold

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-private-entry-unsupported`.

Objective: implement the smallest private CUDA persistent-device entry
scaffold behind the `ChipWorker::run` / `ChipStorageTaskArgs` path without
changing public APIs. The branch may add runtime-owned request/result
plumbing for `persistent_device_uccl_ep_runtime_fusion_entry`, but the normal
`--with-uccl-ep-fused-boundary` outcome must remain `unsupported` unless the
runtime coordinator itself creates a shared dispatch/combine descriptor,
issues the ownership token, records the complete lifetime transition log, and
emits trusted coordinator-owned validation fields.

This is now a narrow implementation slice rather than another docs-only
dependency because PR #152 defined the coordinator boundary and PR #153
defined the private entry contract. The honest implementation floor is still
an unsupported result: creating an entry scaffold is not enough to report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

Required implementation boundaries:

- keep the entry private to the CUDA persistent-device runtime path;
- do not add public `TaskArgs`, public `CallConfig`, or UCCL host-runtime ABI
  fields;
- derive request fields from callable id, `ChipStorageTaskArgs`, private
  rank/device metadata, persistent graph descriptor handles, UCCL-EP
  capability metadata, descriptor allocation policy, validation policy, and a
  runtime-owned output sink;
- emit `unsupported` with explicit failure fields when the coordinator,
  descriptor allocator, UCCL-EP runtime path, or validation policy is absent;
- reject adapter-only provenance, example-side JSON, handoff metadata, public
  `TaskArgs`, and public `CallConfig` as pass-evidence sources;
- keep `persistent_device_uccl_ep_runtime_fusion.status: passed` and
  `actual_fused_cross_gpu_execution: true` unreachable unless real
  coordinator-owned evidence exists.

Implemented scaffold surface:

- `src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h` defines the
  private `PtoCudaRuntimeFusionRequest`,
  `PtoCudaRuntimeFusionResult`, status values, failure bits, forbidden
  evidence source values, and
  `persistent_device_uccl_ep_runtime_fusion_entry`;
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp` builds the private
  request on the persistent DAG path from callable id, graph descriptor,
  private CUDA communication descriptor when configured, and a runtime-owned
  result sink;
- `tests/ut/py/test_cuda_runtime_fusion_private_entry.py` compiles the
  private header in isolation and verifies unsupported missing-surface
  behavior, forbidden pass-evidence rejection, and absence from the common
  runtime C API.

Allowed scope:

- private CUDA persistent-device runtime entry scaffolding;
- focused unit tests for unsupported private-entry behavior and forbidden
  evidence paths;
- review-artifact docs and tests that pin the unsupported evidence boundary;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency work.

PR #155 met this objective as a private unsupported scaffold only. It added
the private request/result ABI and a CUDA persistent DAG host-runtime hook
that records an unsupported result from private state currently available in
the runtime path. It did not implement a coordinator, descriptor allocator,
UCCL-EP runtime path, validation policy, UCCL-EP capability metadata,
`ChipStorageTaskArgs` request materialization, pass evidence, or fresh H200
fused success.

## Closed Invalid ChipStorageTaskArgs Request Boundary Attempt

Closed branch:
`nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request`.

PR #157 (`Thread private ChipStorageTaskArgs request`) is closed invalid and
must not be resurrected. The implementation assigned the persistent DAG run
`args` pointer to `PtoCudaRuntimeFusionRequest::chip_storage_task_args` and
recorded `sizeof(ChipStorageTaskArgs)`, but that pointer is a
`PtoCudaPersistentDagArgs *` inside
`src/cuda/platform/onboard/host/pto_runtime_c_api.cpp`, not a
`ChipStorageTaskArgs *` materialized by `ChipWorker::run`.

Before the private-request-envelope dependency, the CUDA code state was
unclaimed: no real `ChipStorageTaskArgs` request path reached
`persistent_device_uccl_ep_runtime_fusion_entry`. The persistent DAG scaffold
still records `missing_chip_storage_task_args` when no valid private request
input exists.

## Accepted Private Request Envelope Dependency Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-private-request-envelope`.

Objective: define the private ABI/envelope dependency surface and make the
`ChipWorker::run` handoff honest without expanding public `TaskArgs`, public
`CallConfig`, the common runtime C API, or UCCL host-runtime ABI fields. The
envelope must keep runtime-specific persistent DAG inputs separate from the
chip-storage task-argument pointer and size so `PtoCudaPersistentDagArgs *`
cannot be mislabeled as `ChipStorageTaskArgs *`. Until a real
runtime-specific args pointer is available at the `ChipWorker` boundary, the
typed-args path must reject the private envelope instead of fabricating it.

Required boundaries:

- keep the envelope private to CUDA host-runtime request construction and the
  `ChipWorker::run` rejection boundary;
- do not add public `TaskArgs`, public `CallConfig`, or UCCL host-runtime ABI
  fields;
- carry a real `ChipStorageTaskArgs` pointer/size only when a valid private
  envelope exists, or explicitly reject the path with focused local coverage;
- preserve `PtoCudaPersistentDagArgs *` as persistent DAG runtime input only,
  never as `chip_storage_task_args`;
- keep missing coordinator, descriptor allocator, UCCL-EP runtime path,
  validation policy, UCCL-EP capability metadata, and pass evidence as
  unsupported or failed states;
- keep `persistent_device_uccl_ep_runtime_fusion.status: passed` and
  `actual_fused_cross_gpu_execution: true` unreachable.

Implemented surface in this branch:

- `src/cuda/platform/include/host/pto_cuda_private_run_envelope.h` defines the
  private CUDA run envelope with separate runtime-task-args and typed
  `ChipStorageTaskArgs` fields;
- `src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h` now types
  `PtoCudaRuntimeFusionRequest::chip_storage_task_args` as
  `const ChipStorageTaskArgs *`;
- `src/common/worker/chip_worker.cpp` probes the optional CUDA-only
  `run_prepared_with_cuda_private_args` symbol and explicitly rejects the
  typed-args private-envelope path because it cannot provide runtime-specific
  CUDA args;
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp` unwraps the private
  envelope, keeps `PtoCudaPersistentDagArgs *` as runtime-specific DAG input,
  and copies only `envelope->chip_storage_task_args` into the runtime-fusion
  request;
- `tests/ut/py/test_cuda_runtime_fusion_private_entry.py` compiles the typed
  envelope, guards the source-level private hook, and checks that
  `ChipWorker::run` does not pass `ChipStorageTaskArgs *` as runtime args.

This branch still does not implement the runtime-fusion coordinator,
descriptor allocator, UCCL-EP runtime path, validation policy, UCCL-EP
capability metadata, or pass evidence. The review-safe result therefore
remains unsupported or failed, never
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

PR #160 met this objective as a private request-envelope and host-runtime
handoff dependency only. It added the private run envelope and host-runtime
hook, and it made the `ChipWorker::run` typed-args path reject private
envelope use unless a real runtime-specific CUDA args pointer exists. It did
not implement the runtime-fusion coordinator, descriptor allocator, UCCL-EP
runtime path, validation policy, UCCL-EP capability metadata, or pass
evidence. It did not expand public `TaskArgs`, public `CallConfig`, the
common runtime C API, or UCCL host-runtime ABI fields.

## Accepted Runtime Args Handoff Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-runtime-args-handoff-map`.

Objective: map the next private dependency boundary before another runtime
implementation attempt. The slice should describe how a real
runtime-specific `PtoCudaPersistentDagArgs *` can be associated with a real
`ChipStorageTaskArgs *` at the private CUDA host-runtime handoff while
preserving the separation introduced by PR #160.

This is intentionally a conservative dependency/status slice. The selected
map is not another attempt to have `ChipWorker::run` synthesize runtime args.
The private association point must sit inside the CUDA host-runtime handoff
that already has both sides of the data:

- `ChipWorker::run` owns the real `ChipStorageTaskArgs` materialized from the
  decoded mailbox `TaskArgs` view. That pointer remains typed as
  `const ChipStorageTaskArgs *` and is never relabeled as runtime-specific
  CUDA args.
- The CUDA persistent DAG host-runtime path owns the real
  `PtoCudaPersistentDagArgs *` after it resolves the prepared persistent DAG
  callable and builds the runtime launch request.
- `PtoCudaPrivateRunArgsEnvelope` remains the private association carrier:
  `runtime_task_args` / `runtime_task_args_size` hold the
  `PtoCudaPersistentDagArgs *` view, while `chip_storage_task_args` /
  `chip_storage_task_args_size` hold the `ChipStorageTaskArgs *` view.
- The future implementation owner is the CUDA host runtime, not public
  Python, example JSON, `TaskArgs`, `CallConfig`, the common runtime C API,
  or UCCL host-runtime ABI. It may create the private envelope only after
  both pointers are real and same-invocation.
- `ChipWorker::run` may pass the chip-storage pointer to a private CUDA hook,
  but it must continue to reject any path that asks it to invent
  `PtoCudaPersistentDagArgs *`.

The first implementation after this map should be a narrow private
host-runtime handoff that proves the two pointers are associated inside one
CUDA persistent DAG invocation, with local tests for null, wrong-size,
mismatched-callable, stale-envelope, cross-invocation, and forbidden
public/API evidence-path cases. It still must keep missing coordinator,
descriptor allocator, UCCL-EP runtime path, validation policy, UCCL-EP
capability metadata, and pass evidence as unsupported or failed states.
It must not expand public `TaskArgs`, public `CallConfig`, the common runtime
C API, or UCCL host-runtime ABI fields. It must keep
`persistent_device_uccl_ep_runtime_fusion.status: passed` and
`actual_fused_cross_gpu_execution: true` unreachable.

Allowed scope:

- `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`
- `docs/in_progress/nvidia_backend/communication_selection.md`
- `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`
- `docs/in_progress/nvidia_backend/pr_slicing_plan.md`
- `docs/in_progress/nvidia_backend/dispatch_log.md`
- focused review-artifact tests if assertions need to pin the handoff map

Required non-claims:

- no runtime-fusion success;
- no fresh H200 fused-success evidence;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency
  claim;
- no public `TaskArgs`, public `CallConfig`, common runtime C API, or UCCL
  host-runtime ABI expansion.

PR #162 met this objective as a runtime-args handoff map only. It did not
implement the private association, change CUDA runtime behavior, change the
fused-boundary result shape, claim fresh H200 fused success, report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, or set
`actual_fused_cross_gpu_execution: true`.

## Accepted Private Host Runtime Handoff Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff`.

Objective: implement only the private CUDA persistent DAG host-runtime
handoff needed to associate real same-invocation `ChipStorageTaskArgs *` and
`PtoCudaPersistentDagArgs *` pointers before
`persistent_device_uccl_ep_runtime_fusion_entry` is requested. This is the
narrowest implementation step after PR #162 because the map already selected
the CUDA host runtime as the owner of the association.

Required implementation boundaries:

- keep the association private to the CUDA persistent DAG host-runtime path;
- do not add public `TaskArgs`, public `CallConfig`, common runtime C API, or
  UCCL host-runtime ABI fields;
- accept `ChipStorageTaskArgs *` only when it comes from the current
  `ChipWorker::run` invocation and has the expected size;
- accept `PtoCudaPersistentDagArgs *` only after the prepared persistent DAG
  callable is resolved for that same invocation;
- reject null pointers, wrong sizes, mismatched callable types, stale
  envelopes, cross-invocation envelopes, and forbidden public/API evidence
  paths with focused local tests;
- keep missing coordinator, descriptor allocator, UCCL-EP runtime path,
  validation policy, UCCL-EP capability metadata, and pass evidence as
  unsupported or failed states;
- keep `persistent_device_uccl_ep_runtime_fusion.status: passed` and
  `actual_fused_cross_gpu_execution: true` unreachable unless a later
  coordinator slice emits real fused-boundary evidence.

Allowed future implementation scope:

- private CUDA host-runtime handoff code under the CUDA persistent DAG path;
- focused local tests for the association and rejection cases above;
- review-facing docs/tests that pin the unsupported evidence boundary.

Required future non-claims:

- no runtime-fusion success;
- no fresh H200 fused-success evidence unless the implementation changes the
  fused-boundary result and records a fresh H200 command;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim;
- no public `TaskArgs`, public `CallConfig`, common runtime C API, or UCCL
  host-runtime ABI expansion.

Implemented surface in this branch:

- `src/common/worker/chip_worker.cpp` builds a CUDA-private handoff envelope
  with the real `ChipStorageTaskArgs *`, `sizeof(ChipStorageTaskArgs)`,
  callable id, and a per-`ChipWorker` private invocation id. It does not set
  `runtime_task_args` from that chip-storage pointer.
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp` completes the
  envelope only inside the CUDA persistent DAG host-runtime path, after the
  prepared callable has resolved to `PTO_CUDA_PERSISTENT_OP_DAG_F32_RING`.
  It then validates callable id, invocation id, callable type, and both
  pointer sizes before requesting
  `persistent_device_uccl_ep_runtime_fusion_entry`.
- `src/cuda/platform/include/host/pto_cuda_private_run_envelope.h` now defines
  private envelope status values plus init/validate helpers for null pointer,
  stale envelope, callable mismatch, cross-invocation, callable-type
  mismatch, runtime-args size, and chip-storage size rejection.
- `tests/ut/py/test_cuda_runtime_fusion_private_entry.py` covers the private
  envelope acceptance and rejection cases, the `ChipWorker::run` chip-storage
  handoff, the host-runtime validation point, forbidden pass-evidence
  rejection, and the continued absence of public/common ABI expansion.

This implementation remains a private handoff slice only. It does not add the
runtime-fusion coordinator, descriptor allocator, UCCL-EP runtime path,
validation policy, UCCL-EP capability metadata, pass evidence, H200 fused
success evidence, or any RDMA, multi-node, serving, vLLM, DeepSeek,
throughput, or latency claim. `persistent_device_uccl_ep_runtime_fusion.status:
passed` and `actual_fused_cross_gpu_execution: true` remain unreachable until
a later coordinator slice emits real fused-boundary evidence.

PR #164 met this objective as a private host-runtime handoff only. It merged
as `be914b97898468033c7f834dde0c43466353ac95` and is no longer selected as
future work.

## Accepted UCCL-EP Capability Metadata Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-capability-metadata-map`.

Objective: map the private UCCL-EP capability metadata that the
`persistent_device_uccl_ep_runtime_fusion_entry` request will need before any
coordinator implementation can consume the private host-runtime handoff from
PR #164. This is a conservative docs/test dependency slice because the
coordinator still lacks descriptor allocation, UCCL-EP runtime dispatch,
validation policy, pass evidence, and H200 fused-success evidence.

Required dependency boundaries:

- keep UCCL-EP capability metadata private to the CUDA persistent-device
  runtime path and chip-child private metadata;
- identify the minimum fields needed by the coordinator request: capability
  id, world size, rank-to-device map, descriptor vocabulary, transport mode,
  adapter provenance handles, and setup/validation failure ownership;
- define how missing, stale, mismatched-rank, mismatched-world-size, or
  public/API-sourced capability metadata is reported as unsupported or failed;
- preserve the PR #164 association between real same-invocation
  `ChipStorageTaskArgs *` and `PtoCudaPersistentDagArgs *` pointers;
- keep public `TaskArgs`, public `CallConfig`, the common runtime C API,
  UCCL host-runtime ABI fields, example JSON, adapter provenance, and handoff
  metadata as forbidden pass-evidence paths.

Implemented surface in this branch:

- maps private UCCL-EP capability metadata as a dependency vocabulary only;
- keeps the metadata private to the CUDA persistent-device runtime path and
  chip-child private metadata;
- records capability id, world size, rank-to-device map, descriptor
  vocabulary, transport mode, adapter provenance handles, and
  setup/validation failure ownership as the minimum later coordinator fields;
- preserves the PR #164 association between real same-invocation
  `ChipStorageTaskArgs *` and `PtoCudaPersistentDagArgs *`;
- treats missing, stale, mismatched-rank, mismatched-world-size, and
  public/API-sourced capability metadata as unsupported or failed states;
- keeps forbidden pass-evidence paths explicit: public `TaskArgs`, public
  `CallConfig`, common runtime C API, UCCL host-runtime ABI, example JSON,
  adapter provenance, and handoff metadata.

Allowed scope:

- review-facing docs/tests that map the capability metadata dependency;
- focused review-artifact assertions that pin PR #164 acceptance and this
  next selected branch;
- no CUDA runtime behavior change.

Required non-claims:

- no runtime-fusion coordinator implementation;
- no descriptor allocator implementation;
- no UCCL-EP runtime path implementation;
- no validation policy implementation;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim;
- no public `TaskArgs`, public `CallConfig`, common runtime C API, or UCCL
  host-runtime ABI expansion.

PR #166 met this objective as a private capability metadata dependency map
only. It merged as `42b996666e279024b43f490a310c490a591a897d` and is no
longer selected as future work.

## Accepted Validation Policy Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-validation-policy-map`.

Objective: map the private validation policy that a later
`persistent_device_uccl_ep_runtime_fusion_entry` coordinator request must use
after PR #166 defined the private UCCL-EP capability metadata vocabulary.
This is the narrowest missing dependency after capability metadata because it
can define failure ownership without allocating descriptors, implementing the
UCCL-EP runtime path, constructing the coordinator, or claiming pass evidence.

Required dependency boundaries:

- keep validation policy private to the CUDA persistent-device runtime path;
- validate PR #164 same-invocation request args and PR #166 capability
  metadata together before any coordinator can consume them;
- define required failures for missing, stale, mismatched-rank,
  mismatched-world-size, descriptor-vocabulary mismatch, transport-mode
  mismatch, adapter-provenance mismatch, and public/API-sourced metadata;
- keep descriptor allocation policy, UCCL-EP runtime dispatch, coordinator
  implementation, pass evidence, and H200 fused-success evidence out of this
  slice;
- keep public `TaskArgs`, public `CallConfig`, the common runtime C API,
  UCCL host-runtime ABI fields, example JSON, adapter provenance, and handoff
  metadata as forbidden pass-evidence paths.

Implemented surface in this branch:

- the validation policy remains private to the CUDA persistent-device runtime
  path;
- it validates PR #164 same-invocation request args and PR #166 capability
  metadata together before a coordinator can consume either dependency;
- failure ownership is explicit: missing metadata is unsupported, stale
  metadata is failed, mismatched-rank metadata is failed, and
  mismatched-world-size metadata is failed;
- descriptor-vocabulary mismatch is failed because descriptor vocabulary must
  match dispatch/combine payload terms;
- transport-mode mismatch is failed because transport mode must be `ep`;
- adapter-provenance mismatch is failed because adapter provenance handles
  must match the private capability id, invocation id, and rank/device map;
- public/API-sourced metadata is failed as fabricated or untrusted pass
  evidence;
- no descriptor allocation policy implementation, UCCL-EP runtime dispatch,
  coordinator implementation, pass evidence, or H200 fused-success evidence
  is added.

Required non-claims:

- no CUDA runtime behavior change;
- no runtime-fusion coordinator implementation;
- no descriptor allocator implementation;
- no UCCL-EP runtime path implementation;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim.

PR #168 met this objective as a private validation policy dependency map
only. It merged as `e33d232deccdf947b9c382a3605191d0d5ae0004` and is no
longer selected as future work.

## Selected Descriptor Allocation Policy Map Slice

Selected branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map`.

Objective: map the private descriptor allocation policy that a later
`persistent_device_uccl_ep_runtime_fusion_entry` coordinator request must use
after PR #168 defined validation policy. This is the narrowest missing
dependency after validation policy because descriptor allocation policy must
define what the coordinator may allocate before UCCL-EP runtime dispatch,
coordinator implementation, pass evidence, or H200 fused-success evidence can
be claimed.

Required dependency boundaries:

- keep descriptor allocation policy private to the CUDA persistent-device
  runtime path;
- preserve PR #164 same-invocation request args, PR #166 UCCL-EP capability
  metadata, and PR #168 validation policy as prerequisites rather than pass
  evidence;
- define host-control record, device-visible descriptor buffer, dispatch
  descriptor identity, combine descriptor identity, shared-token requirement,
  allocator owner, and allocation lifetime failure ownership;
- define unsupported or failed states: missing policy is unsupported, stale
  policy is failed, non-runtime-owned allocation is failed,
  descriptor-vocabulary mismatch is failed, token-sharing mismatch is failed,
  rank/device mismatch is failed, and public/API-sourced policy fields are
  failed;
- keep UCCL-EP runtime dispatch, coordinator implementation, pass evidence,
  and H200 fused-success evidence out of this slice;
- keep public `TaskArgs`, public `CallConfig`, the common runtime C API,
  UCCL host-runtime ABI fields, example JSON, adapter provenance, and handoff
  metadata as forbidden pass-evidence paths.

Required non-claims:

- no CUDA runtime behavior change;
- no runtime-fusion coordinator implementation;
- no UCCL-EP runtime path implementation;
- no pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim.
