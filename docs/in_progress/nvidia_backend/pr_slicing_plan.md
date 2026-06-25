# Restart PR Slicing Plan

This plan tracks the current PR-sized route for continuing the NVIDIA backend
restart. The dirty orphan workspace is historical input only; new work starts
from `main` and lands through focused GitHub PRs.

## Current Baseline

- Base branch: `main`.
- Current accepted `main`: `9e338948f90fdc4fb13a527159060b2510e12838`,
  after PR #187
  (`Refresh runtime dispatch driver scaffold status`).
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
- PR #170 accepted only the private descriptor allocation policy dependency
  map: allocator owner, host-control record policy, device-visible descriptor
  buffer policy, dispatch/combine descriptor identity, shared-token
  requirement, and allocation lifetime failure ownership. It did not
  implement CUDA runtime behavior, descriptor allocation, UCCL-EP runtime
  dispatch, a coordinator, pass evidence, or H200 fused-success evidence.
- PR #172 accepted only the private UCCL-EP runtime path dependency map:
  runtime-path owner, dispatch descriptor handoff, combine descriptor handoff,
  descriptor-token checks, rank/device checks, transport-mode checks, and
  runtime-path failure ownership. It did not implement CUDA runtime behavior,
  UCCL-EP runtime dispatch, a coordinator, descriptor allocation, pass
  evidence, or H200 fused-success evidence.
- PR #174 accepted only the private UCCL-EP runtime path scaffold:
  `PtoCudaUcclEpRuntimePath`, `PtoCudaUcclEpRuntimeDescriptorView`, private
  descriptor-view validation, and invocation-id propagation through private
  CUDA runtime-fusion request state. It did not implement the runtime-fusion
  coordinator, descriptor allocation, UCCL-EP runtime dispatch, pass evidence,
  fresh H200 fused-success evidence, public `TaskArgs`, public `CallConfig`,
  common runtime C API fields, UCCL host-runtime ABI fields, serving, vLLM,
  DeepSeek, throughput, or latency evidence.
- PR #176 accepted only the private UCCL-EP runtime-fusion descriptor
  allocation scaffold: private host-control record and device-visible
  dispatch/combine descriptor buffer mechanics bound to the PR #174
  same-invocation runtime-path scaffold. It did not implement coordinator
  construction, UCCL-EP runtime dispatch, pass evidence, fresh H200
  fused-success evidence, public `TaskArgs`, public `CallConfig`, common
  runtime C API fields, UCCL host-runtime ABI fields, examples, stable docs,
  serving, vLLM, DeepSeek, throughput, or latency evidence.
- PR #177 recorded the post-PR176 status refresh and selected exactly one
  next coordinator-construction scaffold/status slice. It did not change CUDA
  runtime behavior, result shape, or fused-execution evidence status.
- PR #178 accepted only the private coordinator scaffold/status surface:
  private coordinator-owned state for one `ChipWorker::run` invocation so the
  accepted descriptor allocation and private runtime path are owned together
  with the same invocation id, unsupported/failure status, and output sink. It
  remains unsupported and does not provide UCCL-EP runtime dispatch,
  scheduler/runtime pass evidence, fresh H200 fused-success evidence, public
  API expansion, examples, stable docs, serving, vLLM, DeepSeek, throughput,
  or latency evidence.
- PR #179 recorded the post-PR #178 status refresh and selected exactly one
  next slice:
  `nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status`. That
  active branch may add only a private UCCL-EP runtime-dispatch
  scaffold/status gate from coordinator-owned state. It must remain narrower
  than scheduler/runtime pass evidence, make no fused-success claim, and
  preserve `persistent_device_uccl_ep_runtime_fusion.status` as unsupported or
  failed until real UCCL-EP dispatch and fresh H200 fused-boundary evidence
  exist.
- PR #180 accepted only the private coordinator-owned runtime-dispatch
  scaffold/status gate: missing gate yields
  `missing_runtime_dispatch_scaffold` and a failed private result; an
  eligible prepared gate remains `unsupported`; output is mirrored to the
  runtime-owned sink. It remains unsupported and does not provide real
  UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, fresh H200
  fused-success evidence, public API expansion, examples, stable docs,
  serving, vLLM, DeepSeek, throughput, or latency evidence.
- PR #181 recorded the post-PR180 status refresh and selected
  `nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map` as
  the next dependency slice. It did not change CUDA runtime behavior, result
  shape, or fused-execution evidence status.
- PR #182 accepted only the private UCCL-EP runtime dispatch request/driver
  handoff dependency map. It defined request owner, driver owner, status
  dependency, failure ownership, unsupported handoff state, and failed
  handoff state for a future private handoff. It did not change CUDA runtime
  behavior, result shape, or fused-execution evidence status.
- PR #183 accepted only the private request/driver handoff scaffold/status
  path: private ABI state under `PtoCudaRuntimeFusionCoordinator`, same
  invocation id, coordinator-owned runtime path/gate, request owner, private
  driver-state pointer, runtime-owned output sink, missing/stale handoff
  driver failure, and a valid handoff that remains `unsupported`. It remains
  unsupported/failed only and does not provide real UCCL-EP dispatch/combine
  work, scheduler/runtime pass evidence, H200 fused success, public API
  expansion, examples, stable docs, serving, vLLM, DeepSeek, throughput, or
  latency evidence.
- PR #185 accepted only the private runtime-dispatch driver status map:
  driver-owned unsupported/failed status vocabulary and failure ownership
  after the PR #183 handoff scaffold/status path. It did not implement a
  driver, real UCCL-EP dispatch/combine work, scheduler/runtime pass
  evidence, fresh H200 fused success, public API expansion, examples, stable
  docs, serving, vLLM, DeepSeek, throughput, or latency evidence.
- PR #186 accepted only private runtime-dispatch driver scaffold/status
  ownership: `PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
  `PtoCudaUcclEpRuntimeDispatchDriverStatus`, driver-owned failure bits,
  prepare helper, and host runtime private call. Valid status remains
  unsupported, and malformed/mismatched produces failed private result. It did
  not implement real UCCL-EP dispatch/combine work, scheduler/runtime pass
  evidence, fresh H200 fused success, public API expansion, examples, stable
  docs, serving, vLLM, DeepSeek, throughput, or latency evidence.
- PR #189 accepted only private runtime-dispatch driver backend
  scaffold/status ownership: `PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`,
  `PtoCudaUcclEpRuntimeDispatchDriverBackendStatus`,
  `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_status_name`, and
  `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`.
  Valid backend scaffold/status remains unsupported, and malformed/mismatched
  backend scaffold state produces failed private result. It did not implement
  real UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, fresh
  H200 fused success, public API expansion, examples, stable docs, serving,
  vLLM, DeepSeek, throughput, or latency evidence.
- PR #190 accepted only the private driver backend request dependency map
  after PR #189. It defined future private backend request ownership,
  backend scaffold/status input, descriptor token and rank/device validation,
  invalid public/provenance sources, unsupported states, and failed states.
  It did not implement real UCCL-EP dispatch/combine work, scheduler/runtime
  pass evidence, fresh H200 fused success, public API expansion, examples,
  stable docs, serving, vLLM, DeepSeek, throughput, or latency evidence.
- PR #191 accepted only private runtime-dispatch driver backend request
  scaffold/status ownership:
  `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus`,
  `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestStatus`,
  `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_request_status_name`,
  and
  `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status`.
  Valid backend request scaffold/status remains unsupported, and
  malformed/mismatched backend request scaffold state produces failed private
  result. It did not implement real UCCL-EP dispatch/combine work,
  scheduler/runtime pass evidence, fresh H200 fused success, public API
  expansion, examples, stable docs, serving, vLLM, DeepSeek, throughput, or
  latency evidence.
- PR #192 accepted only the private driver backend dispatch request
  dependency map after PR #191. It defined future private dispatch request
  ownership, backend request scaffold/status input, descriptor token and
  rank/device validation, invalid public/provenance sources, unsupported
  states, and failed states. It did not implement real UCCL-EP
  dispatch/combine work, scheduler/runtime pass evidence, fresh H200 fused
  success, public API expansion, examples, stable docs, serving, vLLM,
  DeepSeek, throughput, or latency evidence.
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
- PR #170: map UCCL EP descriptor allocation policy.
  - Result: merged as `bd0b59ee8d5afc969020d3aea047aafc9f3152be`.
  - Result type: private descriptor allocation policy dependency map only,
    not runtime behavior, descriptor allocation implementation, UCCL-EP
    runtime dispatch, coordinator behavior, pass evidence, or fused execution
    evidence.
- PR #172: map UCCL EP runtime path.
  - Result: merged as `21b2b32a475dc04e19700115af74510daef70859`.
  - Result type: private UCCL-EP runtime path dependency map only, not
    runtime behavior, UCCL-EP runtime dispatch implementation, coordinator
    behavior, descriptor allocation implementation, pass evidence, or fused
    execution evidence.
- PR #174: add private UCCL EP runtime path scaffold.
  - Result: merged as `3b4b19a04855d27289fb9cdad802fee0c47d8265`.
  - Result type: private UCCL-EP runtime path scaffold only, not coordinator
    behavior, descriptor allocation implementation, UCCL-EP runtime dispatch,
    pass evidence, H200 fused-success evidence, public runtime API expansion,
    serving, vLLM, DeepSeek, or performance evidence.
- PR #176: add private UCCL EP descriptor allocation scaffold.
  - Result: merged as `6e0cecc174ae9db47573c4c0f1698be7accb295c`.
  - Result type: private descriptor allocation scaffold only, not coordinator
    construction, UCCL-EP runtime dispatch, pass evidence, H200 fused-success
    evidence, public runtime API expansion, examples, stable docs, serving,
    vLLM, DeepSeek, or performance evidence.
- PR #178: add private UCCL EP coordinator scaffold.
  - Result: merged as `aea89cc9dea8560602c72f84e5ff6e78ca526434`.
  - Result type: private coordinator scaffold/status surface only, not
    runtime dispatch, pass evidence, H200 fused-success evidence, public
    runtime API expansion, examples, stable docs, serving, vLLM, DeepSeek, or
    performance evidence.
- PR #180: add private UCCL EP runtime dispatch scaffold gate.
  - Result: merged as `dc32c52dfccfd7838f865a11c3d4837e8ee568ba`.
  - Result type: private runtime-dispatch scaffold/status gate only, not real
    UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, H200
    fused-success evidence, public runtime API expansion, examples, stable
    docs, serving, vLLM, DeepSeek, or performance evidence.
- PR #181: refresh NVIDIA status after PR 180.
  - Result: merged as `05457b7dead2f561be22c24c72771add880f4562`.
  - Result type: status/slicing refresh only, not runtime behavior or fused
    execution evidence.
- PR #182: map UCCL EP runtime dispatch handoff.
  - Result: merged as `7c02f131ab5f7ad88481079a1813270a0cc02d3a`.
  - Result type: private request/driver handoff dependency map only, not real
    UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, H200
    fused-success evidence, public runtime API expansion, examples, stable
    docs, serving, vLLM, DeepSeek, or performance evidence.
- PR #183: add private runtime dispatch handoff scaffold.
  - Result: merged as `80b6606282956f38ca6c9a3c52c95d0e5e3a457f`.
  - Result type: private request/driver handoff scaffold/status path only,
    not real UCCL-EP dispatch/combine work, scheduler/runtime pass evidence,
    H200 fused-success evidence, public runtime API expansion, examples,
    stable docs, serving, vLLM, DeepSeek, or performance evidence.

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
After PR #170, the private descriptor allocation policy map is accepted as a
dependency slice only; it does not implement descriptor allocation, UCCL-EP
runtime dispatch, coordinator behavior, pass evidence, or H200 fused-success
evidence. After PR #172, the private UCCL-EP runtime path map is accepted as
a dependency slice only; it does not implement CUDA runtime behavior,
UCCL-EP runtime dispatch, coordinator behavior, descriptor allocation, pass
evidence, or H200 fused-success evidence. After PR #174, the private
UCCL-EP runtime path scaffold is accepted as a narrow private implementation
slice only; it does not implement coordinator behavior, descriptor
allocation, UCCL-EP runtime dispatch, pass evidence, or fresh H200
fused-success evidence. After PR #176, the private descriptor allocation
scaffold is accepted as a narrow private implementation slice only; it does
not implement coordinator construction, UCCL-EP runtime dispatch, pass
evidence, fresh H200 fused-success evidence, public API expansion, examples,
stable docs, serving, vLLM, DeepSeek, throughput, or latency. After PR #178,
the private coordinator scaffold/status surface is accepted only for
coordinator-owned state: accepted descriptor allocation, runtime path, same
invocation id, unsupported/failure status, and output sink for one private
`ChipWorker::run` invocation. It remains unsupported and does not provide
runtime dispatch, pass evidence, or H200 fused success. After PR #180, the
private runtime-dispatch scaffold/status gate is accepted only for
coordinator-owned gate behavior: missing gate yields
`missing_runtime_dispatch_scaffold` and a failed private result; an eligible
prepared gate remains `unsupported`; output is mirrored to the runtime-owned
sink. After PR #181, the current accepted baseline is
`05457b7dead2f561be22c24c72771add880f4562`, and the active slice is exactly
one narrow private UCCL-EP runtime dispatch request/driver handoff map:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map`. This
map remains docs/test dependency evidence only.

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

## Accepted Descriptor Allocation Policy Map Slice

Accepted branch:
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

Implemented surface in this branch:

- the descriptor allocation policy remains private to the CUDA
  persistent-device runtime path;
- PR #164 same-invocation request args, PR #166 UCCL-EP capability metadata,
  and PR #168 validation policy are prerequisites only, never pass evidence;
- the allocator owner is the future private
  `persistent_device_uccl_ep_runtime_fusion` coordinator inside one CUDA
  persistent-device runtime run context;
- host-control record policy defines private per-invocation records for
  invocation id, persistent graph descriptor id, UCCL capability id,
  validated rank/device map, descriptor vocabulary, allocation state, runtime
  owner, and shared ownership token slot;
- device-visible descriptor buffer policy defines future coordinator-owned
  buffers allocated through the CUDA persistent-device runtime allocator and
  visible only to the persistent-device scheduler and UCCL-EP runtime path;
- dispatch descriptor identity is the validated graph descriptor id,
  capability id, invocation id, rank/device map, dispatch vocabulary, payload
  shape, and coordinator-issued shared token;
- combine descriptor identity uses the same validated ids, rank/device map,
  combine vocabulary, payload shape, and the same shared token as dispatch;
- the shared-token requirement is strict: dispatch and combine descriptors
  must carry one coordinator-issued token, and token-sharing mismatch is
  failed;
- allocation lifetime failure ownership belongs to the same private runtime
  owner, which must report stale policy, non-runtime-owned allocation,
  descriptor-vocabulary mismatch, token-sharing mismatch, rank/device
  mismatch, and public/API-sourced policy fields as failed states.

Required non-claims:

- no CUDA runtime behavior change;
- no descriptor allocation implementation;
- no runtime-fusion coordinator implementation;
- no UCCL-EP runtime path implementation;
- no pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim.
- no public `TaskArgs`, public `CallConfig`, common runtime C API, or UCCL
  host-runtime ABI expansion.

PR #170 met this objective as a private descriptor allocation policy
dependency map only. It merged as
`bd0b59ee8d5afc969020d3aea047aafc9f3152be` and is no longer selected as
future work.

## Accepted UCCL-EP Runtime Path Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map`.

Objective: map the private UCCL-EP runtime path that a later
`persistent_device_uccl_ep_runtime_fusion_entry` coordinator request must use
after PR #170 defined descriptor allocation policy. This is the narrowest
missing dependency after descriptor allocation policy because the runtime path
must define how dispatch and combine descriptor views reach UCCL-EP runtime
logic before any implementation, coordinator behavior, pass evidence, or H200
fused-success evidence can be claimed.

Required dependency boundaries:

- keep the UCCL-EP runtime path private to the CUDA persistent-device runtime
  path;
- preserve PR #164 same-invocation request args, PR #166 UCCL-EP capability
  metadata, PR #168 validation policy, and PR #170 descriptor allocation
  policy as prerequisites rather than pass evidence;
- define the runtime-path owner, dispatch descriptor handoff, combine
  descriptor handoff, descriptor-token checks, rank/device checks,
  transport-mode checks, and runtime-path failure ownership;
- define unsupported or failed states: missing runtime path is unsupported,
  stale descriptor views are failed, descriptor-token mismatch is failed,
  rank/device mismatch is failed, transport-mode mismatch is failed,
  descriptor-vocabulary mismatch is failed, and public/API-sourced
  runtime-path fields are failed;
- keep runtime-path implementation, coordinator implementation, pass
  evidence, and H200 fused-success evidence out of this slice;
- keep public `TaskArgs`, public `CallConfig`, the common runtime C API,
  UCCL host-runtime ABI fields, example JSON, adapter provenance, and handoff
  metadata as forbidden pass-evidence paths.

Required non-claims:

- no CUDA runtime behavior change;
- no UCCL-EP runtime path implementation;
- no runtime-fusion coordinator implementation;
- no descriptor allocation implementation;
- no pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim;
- no public `TaskArgs`, public `CallConfig`, common runtime C API, or UCCL
  host-runtime ABI expansion.

Implemented surface in this branch:

- the runtime-path owner is the future private
  `persistent_device_uccl_ep_runtime_fusion` coordinator inside one CUDA
  persistent-device runtime run context;
- the dispatch descriptor handoff consumes the PR #170 dispatch descriptor
  identity: invocation id, persistent graph descriptor id, UCCL capability id,
  validated rank/device map, descriptor vocabulary, dispatch payload shape,
  and coordinator-issued shared token;
- the combine descriptor handoff consumes the PR #170 combine descriptor
  identity with the same invocation id, persistent graph descriptor id, UCCL
  capability id, validated rank/device map, descriptor vocabulary, combine
  payload shape, and exactly the same coordinator-issued shared token;
- descriptor-token checks fail unless dispatch and combine descriptor views
  carry the same coordinator-issued token and the token belongs to the current
  same-invocation request;
- rank/device checks fail unless the persistent graph descriptor, private
  UCCL-EP capability metadata, and Worker-local CUDA device ordering agree;
- transport-mode checks fail unless the private UCCL-EP capability metadata
  declares `transport mode: ep` before either descriptor handoff is consumed;
- runtime-path failure ownership remains private to the future coordinator:
  missing runtime path is unsupported, stale descriptor views are failed,
  descriptor-token mismatch is failed, rank/device mismatch is failed,
  transport-mode mismatch is failed, descriptor-vocabulary mismatch is failed,
  and public/API-sourced runtime-path fields are failed;
- public `TaskArgs`, public `CallConfig`, common runtime C API fields, UCCL
  host-runtime ABI fields, example JSON, adapter provenance, and handoff
  metadata remain forbidden pass-evidence paths for this dependency.

This branch remains a docs/test dependency slice. It does not implement the
runtime path, construct the coordinator, allocate descriptor memory, dispatch
UCCL-EP work, emit pass evidence, or add fresh H200 fused-success evidence.

PR #172 met this objective as a private UCCL-EP runtime path dependency map
only, merged as `21b2b32a475dc04e19700115af74510daef70859`. It did not
implement CUDA runtime behavior, UCCL-EP runtime dispatch, a coordinator,
descriptor allocation, pass evidence, or H200 fused-success evidence.

## Accepted UCCL-EP Runtime Path Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl`.

Objective: implement the narrow private UCCL-EP runtime path scaffold after
PR #172 mapped the required runtime-path owner, descriptor handoffs, token
checks, rank/device checks, transport-mode checks, and failure ownership.
This is the narrowest missing implementation step because the runtime path
must exist before a later coordinator can truthfully route dispatch/combine
descriptor views into UCCL-EP runtime logic.

Required implementation boundaries:

- keep the runtime path private to the CUDA persistent-device runtime path;
- consume only the PR #164 same-invocation request args, PR #166 capability
  metadata, PR #168 validation policy, PR #170 descriptor allocation policy,
  and PR #172 runtime-path map as prerequisites;
- add no public `TaskArgs`, public `CallConfig`, common runtime C API, or
  UCCL host-runtime ABI fields;
- preserve missing descriptor allocation and missing coordinator as
  unsupported or failed states, not pass evidence;
- reject public/API-sourced runtime-path fields, example JSON, adapter
  provenance, and handoff metadata as fabricated or untrusted pass evidence;
- keep descriptor allocation implementation, coordinator implementation,
  pass evidence, and H200 fused-success evidence out of this slice.

Implemented surface in this branch:

- adds the private `PtoCudaUcclEpRuntimePath` and
  `PtoCudaUcclEpRuntimeDescriptorView` scaffold below the CUDA
  persistent-device runtime boundary;
- carries the same-invocation id through `PtoCudaRuntimeFusionRequest` so
  runtime-path descriptor views can be checked against the private run
  envelope;
- validates private runtime-path descriptor handoff fields for stale views,
  descriptor-token mismatch, rank/device mismatch, transport-mode mismatch,
  descriptor-vocabulary mismatch, and public/API-sourced runtime-path fields;
- treats public/API, example JSON, adapter provenance, handoff metadata, and
  payload provenance as fabricated or untrusted pass evidence;
- leaves missing coordinator and missing descriptor allocator as unsupported
  or failed states and keeps the normal result non-passing.

Required non-claims:

- no runtime-fusion coordinator implementation;
- no descriptor allocation implementation;
- no pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim.

Remaining implementation gaps after this slice are the runtime-fusion
coordinator, descriptor allocation, real UCCL-EP runtime dispatch, pass
evidence, and fresh H200 fused-success evidence.

PR #174 met this objective as a private UCCL-EP runtime path scaffold only,
merged as `3b4b19a04855d27289fb9cdad802fee0c47d8265`. It accepted
`PtoCudaUcclEpRuntimePath`, `PtoCudaUcclEpRuntimeDescriptorView`, private
descriptor-view validation, and invocation-id propagation through private
CUDA runtime-fusion request state. It did not implement the runtime-fusion
coordinator, descriptor allocation, UCCL-EP runtime dispatch, pass evidence,
fresh H200 fused-success evidence, public `TaskArgs`, public `CallConfig`,
common runtime C API fields, UCCL host-runtime ABI fields, serving, vLLM,
DeepSeek, throughput, or latency evidence.

## Accepted Descriptor Allocation Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl`.

Objective: implement only the private descriptor allocation mechanics needed
after PR #170 mapped descriptor allocation policy and PR #174 made the
private UCCL-EP runtime path scaffold visible to runtime-fusion request state.
This is narrower than constructing the runtime-fusion coordinator and narrower
than dispatching UCCL-EP runtime work.

Required implementation boundaries:

- keep allocation private to the CUDA persistent-device runtime path;
- allocate or model only the private host-control record and device-visible
  dispatch/combine descriptor buffer required by the PR #170 policy;
- bind allocations to the same invocation id carried by the PR #174
  runtime-path scaffold;
- preserve PR #164 same-invocation request args, PR #166 capability
  metadata, PR #168 validation policy, PR #170 allocation policy, PR #172
  runtime-path map, and PR #174 runtime-path scaffold as prerequisites rather
  than pass evidence;
- keep missing coordinator and missing UCCL-EP runtime dispatch as
  unsupported or failed states;
- keep public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, example JSON, adapter provenance, handoff
  metadata, and payload provenance out of pass-evidence paths.

Required non-claims:

- no runtime-fusion coordinator construction;
- no UCCL-EP runtime dispatch;
- no pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim.

Implemented surface in this branch:

- `src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h` defines
  `PtoCudaUcclEpDescriptorHostControl`,
  `PtoCudaUcclEpDeviceDescriptorBuffer`, and
  `PtoCudaUcclEpDescriptorAllocation` for the private host-control record,
  device-visible dispatch/combine descriptor buffer, and allocation bundle.
- `pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors` constructs the
  private allocation from a `PtoCudaRuntimeFusionRequest`, binding dispatch
  and combine descriptor views to the same invocation id, persistent graph
  descriptor, rank/device map, descriptor vocabulary, and shared token.
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp` invokes that helper
  from `CudaDeviceRunner::record_runtime_fusion_unsupported`, stores the
  allocation in private runner state, and passes the allocation plus PR #174
  runtime path into the private entry request.
- `tests/ut/py/test_cuda_runtime_fusion_private_entry.py` covers the private
  allocation helper and proves the entry remains unsupported, with descriptor
  allocation and UCCL-EP runtime path no longer marked missing but the
  coordinator still missing.

This branch still does not construct
`persistent_device_uccl_ep_runtime_fusion`, does not dispatch UCCL-EP runtime
work, does not expose the allocation through public APIs, and does not change
the accepted fused-boundary evidence state.

PR #176 met this objective as a private descriptor allocation scaffold only,
merged as `6e0cecc174ae9db47573c4c0f1698be7accb295c`. It accepted the
private host-control record, device-visible dispatch/combine descriptor
buffer mechanics, allocation bundle, same-invocation binding, and private
runtime-path handoff into request state. It did not implement coordinator
construction, UCCL-EP runtime dispatch, pass evidence, fresh H200
fused-success evidence, public `TaskArgs`, public `CallConfig`, common
runtime C API fields, UCCL host-runtime ABI fields, examples, stable docs,
serving, vLLM, DeepSeek, throughput, or latency evidence.

## Accepted Runtime Fusion Coordinator Scaffold Status Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status`.

Objective: implement only the private
runtime-fusion-coordinator scaffold/status slice after PR #176 and PR #177.
The slice defines and wires private coordinator state needed to own the
accepted descriptor allocation and private runtime path, but it stays narrower
than UCCL-EP runtime dispatch and narrower than pass evidence.

Implemented boundaries:

- keep coordinator construction private to the CUDA persistent-device runtime
  path for one `ChipWorker::run` invocation;
- consume PR #164 same-invocation request args, PR #166 capability metadata,
  PR #168 validation policy, PR #170 allocation policy, PR #172 runtime-path
  map, PR #174 runtime-path scaffold, and PR #176 descriptor allocation
  scaffold as prerequisites rather than pass evidence;
- define or wire only private coordinator state needed to own the descriptor
  allocation, runtime path, unsupported/failure status, and output sink;
- clear `missing_coordinator` only when the request points at
  coordinator-owned descriptor allocation and runtime path state;
- leave UCCL-EP runtime dispatch, scheduler/runtime pass evidence, and fresh
  H200 fused-success evidence out of scope;
- keep public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, examples, stable docs, adapter provenance,
  handoff metadata, and payload provenance out of pass-evidence paths.

Implemented surface in this branch:

- `PTO_CUDA_RUNTIME_FUSION_COORDINATOR_VERSION`;
- `PtoCudaRuntimeFusionCoordinator`;
- `pto_cuda_runtime_fusion_prepare_private_coordinator`;
- `pto_cuda_runtime_fusion_request_has_private_coordinator_shape`;
- `pto_cuda_runtime_fusion_validate_private_coordinator`;
- `CudaDeviceRunner::record_runtime_fusion_unsupported` now records
  `runtime_fusion_coordinator_`, sets `request.coordinator`, and passes the
  coordinator-owned descriptor allocation, runtime path, and output sink to
  the private runtime-fusion entry.

Required non-claims for this slice:

- no UCCL-EP runtime dispatch;
- no pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no public API expansion, examples, stable docs, RDMA, multi-node transport,
  serving, vLLM, DeepSeek, throughput, or latency claim.

PR #178 accepted this coordinator scaffold/status surface only, merged as
`aea89cc9dea8560602c72f84e5ff6e78ca526434`. It accepted private
coordinator-owned state for one `ChipWorker::run` invocation: accepted
descriptor allocation, runtime path, same invocation id, unsupported/failure
status, and output sink. It remains unsupported and did not implement
UCCL-EP runtime dispatch, scheduler/runtime pass evidence, fresh H200
fused-success evidence, public `TaskArgs`, public `CallConfig`, common
runtime C API fields, UCCL host-runtime ABI fields, examples, stable docs,
serving, vLLM, DeepSeek, throughput, or latency.

## Accepted Runtime Dispatch Scaffold Status Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status`.

Objective: add only a private UCCL-EP runtime-dispatch scaffold/status gate
from coordinator-owned state after PR #178. The slice may consume the
coordinator-owned descriptor allocation and runtime path, verify dispatch
eligibility for one private invocation, and record explicit unsupported or
failed status in the runtime-owned output sink.

This branch implements the private status gate only. It adds coordinator-owned
dispatch-scaffold eligibility state and validation so a private invocation
with PR #178 descriptor allocation/runtime path can be distinguished from one
that lacks the scaffold/status gate. A missing gate is reported as an explicit
failed private result with `missing_runtime_dispatch_scaffold`; an eligible
gate remains `unsupported`.

Required boundaries:

- keep the gate private to the CUDA persistent-device runtime path;
- consume PR #178 coordinator-owned state as a prerequisite rather than pass
  evidence;
- do not run real UCCL-EP dispatch/combine work;
- keep scheduler/runtime pass evidence and fresh H200 fused-success evidence
  out of scope;
- keep public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, examples, stable docs, adapter provenance,
  handoff metadata, and payload provenance out of pass-evidence paths.

Required non-claims:

- no UCCL-EP runtime dispatch success;
- no scheduler/runtime pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no public API expansion, examples, stable docs, RDMA, multi-node transport,
  serving, vLLM, DeepSeek, throughput, or latency claim.

PR #180 accepted this private runtime-dispatch scaffold/status gate only,
merged as `dc32c52dfccfd7838f865a11c3d4837e8ee568ba`. Missing gate state
yields `missing_runtime_dispatch_scaffold` and a failed private result; an
eligible prepared gate remains `unsupported`; output is mirrored to the
runtime-owned sink. It did not implement real UCCL-EP dispatch/combine work,
scheduler/runtime pass evidence, fresh H200 fused-success evidence, public
`TaskArgs`, public `CallConfig`, common runtime C API fields, UCCL
host-runtime ABI fields, examples, stable docs, serving, vLLM, DeepSeek,
throughput, or latency.

## Runtime Dispatch Request Handoff Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map`.

Objective: define only the private UCCL-EP runtime dispatch request/driver
handoff map after PR #181. This dependency slice maps how the
coordinator-owned scaffold/status gate hands a validated request to a future
runtime driver, which component owns the request, which component owns the
driver state, and which unsupported or failed states block handoff before any
real dispatch work can run.

This slice starts from PR #181 at
`05457b7dead2f561be22c24c72771add880f4562`. PR #180 remains the status
dependency for the handoff: missing gate yields
`missing_runtime_dispatch_scaffold` and a failed private result; an eligible
prepared gate remains `unsupported`; output is mirrored to the runtime-owned
sink.

Mapped ownership:

- request owner: the private `PtoCudaRuntimeFusionCoordinator`, using only
  coordinator-owned descriptor allocation, private runtime path, validation
  policy, capability metadata, invocation id, PR #180 scaffold/status gate,
  and runtime-owned output sink;
- driver owner: a future private UCCL-EP runtime dispatch driver below the
  CUDA persistent-device runtime path, not public TaskArgs, public
  CallConfig, common runtime C API, or UCCL host-runtime ABI state;
- status dependency: the PR #180 runtime-dispatch scaffold/status gate must
  be present and prepared before any request can be handed to a driver;
- failure ownership: the coordinator owns handoff failures until a later
  private driver scaffold accepts and records driver-owned status.

Unsupported handoff state covers absent prepared gate, missing request
fields, or missing private driver. Failed handoff state covers stale
invocation id, rank/device mismatch, descriptor-token mismatch, failed
scaffold/status gate, public/API-sourced handoff fields, or fabricated pass
evidence.

Required boundaries:

- keep the map private to the CUDA persistent-device runtime path;
- consume PR #180 scaffold/status gate state as a prerequisite rather than
  pass evidence;
- define request owner, driver owner, status dependency, and failure
  ownership for the future handoff;
- no UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public TaskArgs;
- no public CallConfig;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples;
- no stable docs;
- no performance claims;
- keep adapter provenance, handoff metadata, and payload provenance out of
  pass-evidence paths.

Required non-claims:

- no UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no public API expansion, examples, stable docs, RDMA, multi-node transport,
  serving, vLLM, DeepSeek, throughput, or latency claim.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status`.
This is exactly one next PR-sized implementation slice. It may implement only
a private request/driver handoff scaffold/status path for the map above. It
must stay narrower than pass evidence, return only `unsupported` or `failed`
states, and must not run real UCCL-EP dispatch/combine work.

## Accepted Runtime Dispatch Request Handoff Scaffold Status Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status`.

PR #183 merged as `80b6606282956f38ca6c9a3c52c95d0e5e3a457f` and accepted
only the private UCCL-EP runtime dispatch request/driver handoff
scaffold/status path after PR #182. This slice started from
`7c02f131ab5f7ad88481079a1813270a0cc02d3a` and consumes the PR #180
coordinator-owned runtime-dispatch scaffold/status gate plus the PR #182
request/driver handoff map. It remains narrower than pass evidence and does
not run real UCCL-EP dispatch/combine work.

Implemented surface in this branch:

- adds `PtoCudaUcclEpRuntimeDispatchRequestHandoffScaffoldStatus` and
  `PtoCudaUcclEpRuntimeDispatchHandoffDriverState` as private ABI state under
  `PtoCudaRuntimeFusionCoordinator`;
- prepares the handoff scaffold only from the same invocation id,
  coordinator-owned runtime path, PR #180 prepared gate, and runtime-owned
  output sink;
- validates request owner, driver-state pointer, same invocation id,
  coordinator-owned gate, and private driver placeholder state;
- records a missing or stale handoff driver as
  `missing_runtime_dispatch_handoff_driver` with a failed private result;
- records a valid scaffold/status handoff as `unsupported` with
  `unsupported_boundary`;
- wires the scaffold only in the existing private CUDA host-runtime
  unsupported path.

Required boundaries:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public TaskArgs;
- no public CallConfig;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples;
- no stable docs;
- no performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map`. This is
exactly one next PR-sized docs/test dependency slice. It may map only the
private driver-owned unsupported/failed status vocabulary and failure
ownership after this handoff scaffold, still without real UCCL-EP
dispatch/combine work, pass evidence, or H200 fused-success claims.

## Post-PR183 Status Refresh Slice

Branch:
`nvidia-goal-status-post-runtime-dispatch-handoff-scaffold`.

Objective: record PR #183 as accepted only for the private request/driver
handoff scaffold/status path and keep the next slice selection at
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map`. This
status refresh changes only review-facing docs/tests. It does not edit CUDA
runtime/source files, run real UCCL-EP dispatch/combine work, provide
scheduler/runtime pass evidence, or claim H200 fused success.

## Runtime Dispatch Driver Status Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map`.

Objective: define only the private driver-owned unsupported/failed status
vocabulary and failure ownership after PR #183. PR #183 merged as
`80b6606282956f38ca6c9a3c52c95d0e5e3a457f` and accepted only the
request/driver handoff scaffold/status path. This slice does not implement a
driver and does not run real UCCL-EP dispatch/combine work.

The private driver owner is the future UCCL-EP runtime dispatch driver below
the CUDA persistent-device runtime path. The failure owner boundary is:
missing driver remains handoff-owned failed until a driver accepts the
handoff; stale accepted driver is driver-owned failed after that acceptance;
valid handoff remains `unsupported` until real dispatch and combine are
implemented.

Unsupported driver vocabulary:

- `driver_missing`;
- `driver_stale`;
- `driver_not_bound_to_handoff`;
- `driver_no_dispatch_backend`;
- `driver_no_combine_backend`;
- `driver_unsupported_boundary`.

Failed driver vocabulary:

- `driver_owner_mismatch`;
- `driver_invocation_mismatch`;
- `driver_runtime_path_mismatch`;
- `driver_descriptor_token_mismatch`;
- `driver_rank_device_mismatch`;
- `driver_status_sink_mismatch`;
- `driver_public_api_sourced_state`;
- `driver_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-scaffold-status`.
This is exactly one next PR-sized implementation slice. It may add only a
private driver scaffold/status owner for the vocabulary above, and still must
not run real UCCL-EP dispatch/combine work or claim pass evidence.

## Runtime Dispatch Driver Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-scaffold-status`.

Objective: implement only private driver scaffold/status ownership for the
PR #185 mapped vocabulary after PR #185 merged as
`8619767d0eacb5c870b6a56337c6bcb380a2af75`. The private code surface is
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`,
and the driver-owned failure bits in
`src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h`.

The valid prepared driver scaffold is bound to the PR #183 handoff status,
private handoff driver state, coordinator-owned runtime path, runtime-owned
output sink, and same invocation id. It preserves unsupported behavior:
`persistent_device_uccl_ep_runtime_fusion.status` is not `passed`,
`actual_fused_cross_gpu_execution` remains false, and no dispatch/combine
backend is claimed.

Malformed/stale/mismatched private driver scaffold/status produces a failed
private result using driver-owned vocabulary:
`driver_owner_mismatch`, `driver_invocation_mismatch`,
`driver_runtime_path_mismatch`, `driver_descriptor_token_mismatch`,
`driver_rank_device_mismatch`, `driver_status_sink_mismatch`,
`driver_public_api_sourced_state`, and
`driver_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-goal-status-post-runtime-dispatch-driver-scaffold-status`. This is a
review-facing status refresh only.

## Post-Runtime-Dispatch-Driver-Scaffold Status Refresh

Branch:
`nvidia-goal-status-post-runtime-dispatch-driver-scaffold-status`.

Objective: record PR #186 merged as
`7589e2df44ad4df9c200cd4ec673dacac0a27a71`
(`Add runtime dispatch driver scaffold status`) as accepted only for private
runtime-dispatch driver scaffold/status ownership. The accepted private
surface is `PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`, driver-owned failure bits,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`,
and the host runtime private call.

The valid status remains unsupported with `driver_unsupported_boundary`.
Malformed/mismatched produces failed private result through driver-owned
failure names, including `driver_owner_mismatch`,
`driver_invocation_mismatch`, `driver_runtime_path_mismatch`,
`driver_descriptor_token_mismatch`, `driver_rank_device_mismatch`,
`driver_status_sink_mismatch`, `driver_public_api_sourced_state`, and
`driver_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-map`.
This is selected exactly one next PR-sized dependency map slice for real
runtime dispatch driver request/backend ownership. It is not
implementation/pass evidence and remains narrower than real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, or H200 fused-success
claims.

## Runtime Dispatch Driver Backend Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-map`.

Objective: map the future private runtime dispatch driver's request/backend
ownership boundary after PR #186. The slice may define how the driver request,
dispatch backend, combine backend, status sink, and driver-owned failure
ownership fit together, but must not implement real UCCL-EP dispatch/combine
work, run H200 fused evidence, or report pass evidence.

This map follows PR #186, which accepted only
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`, driver-owned failure bits, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`.
The valid prepared driver scaffold remains `unsupported`; it is not a
backend request, backend execution result, or pass-evidence source.

Boundary ownership:

- private driver request owner: the future private runtime dispatch driver
  accepts the PR #186 scaffold only after the coordinator-owned handoff,
  invocation id, runtime path, descriptor token, rank/device map, and
  runtime-owned output sink match;
- dispatch backend placeholder: a private driver-owned placeholder for the
  future UCCL-EP dispatch backend, with no transport calls, payload transfer,
  kernel launch, scheduler transition, or pass evidence;
- combine backend placeholder: a private driver-owned placeholder for the
  future UCCL-EP combine backend, with no reduce/combine transport, payload
  release, kernel launch, scheduler transition, or pass evidence;
- status sink owner: the runtime-owned output sink remains the only sink for
  review-facing status fields; the private driver may record driver status
  and failure names there but does not own example JSON, adapter-only
  provenance, public `TaskArgs`, public `CallConfig`, common runtime C API, or
  UCCL host-runtime ABI state;
- driver-owned failure propagation: once the driver accepts the valid
  scaffold, backend request/backend/status-sink mismatches become
  driver-owned failed states instead of coordinator-owned unsupported states.

Unsupported backend-map states:

- `driver_backend_request_unbound`;
- `driver_dispatch_backend_placeholder`;
- `driver_combine_backend_placeholder`;
- `driver_status_sink_unbound`;
- `driver_backend_map_unsupported_boundary`.

Failed backend-map states:

- `driver_backend_owner_mismatch`;
- `driver_backend_invocation_mismatch`;
- `driver_backend_runtime_path_mismatch`;
- `driver_backend_descriptor_token_mismatch`;
- `driver_backend_rank_device_mismatch`;
- `driver_backend_status_sink_mismatch`;
- `driver_backend_public_api_sourced_state`;
- `driver_backend_fabricated_pass_evidence`.

The invalid pass-evidence boundary rejects example JSON, adapter-only
provenance, public `TaskArgs`, public `CallConfig`, common runtime C API
fields, UCCL host-runtime ABI fields, and hand-authored review artifacts as
evidence for the driver request/backend boundary. If any of those surfaces
supply pass-like backend data, the map treats it as
`driver_backend_fabricated_pass_evidence` or
`driver_backend_public_api_sourced_state`, not fused execution evidence.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
driver backend scaffold/status only. It must remain narrower than real
UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, H200 fused
success, public API expansion, examples, stable docs, or performance claims.

## Runtime Dispatch Driver Backend Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-scaffold-status`.

PR:
`#189` <https://github.com/uv-xiao/pto-cu/pull/189>.

Objective: implement the private driver backend scaffold/status layer selected
by the PR #188 backend map after
`7bc598f75d5738193a7b53fa10a751f2518edb17`. The slice adds only
header-local private ABI plumbing and private-entry coverage for the future
runtime dispatch driver's backend owner. It does not implement real UCCL-EP
dispatch/combine work.

Implementation evidence:

- `src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h` adds
  `PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`;
- the same header adds
  `PtoCudaUcclEpRuntimeDispatchDriverBackendStatus`;
- the same header adds
  `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_status_name`;
- the same header adds
  `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`;
- the same header adds
  `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD`;
- `tests/ut/py/test_cuda_runtime_fusion_private_entry.py` adds
  `test_private_runtime_dispatch_driver_backend_scaffold_status_is_driver_owned`.

The valid prepared backend scaffold/status remains `unsupported`,
`actual_fused_cross_gpu_execution` remains `0`, and no passed status or
transport/backend execution is reported. The private backend scaffold records
driver backend ownership, invocation id, runtime path, status sink, descriptor
token, and rank/device/world metadata only.

Unsupported backend-scaffold states:

- `driver_backend_request_unbound`;
- `driver_dispatch_backend_placeholder`;
- `driver_combine_backend_placeholder`;
- `driver_status_sink_unbound`;
- `driver_backend_map_unsupported_boundary`.

Failed backend-scaffold states:

- `driver_backend_owner_mismatch`;
- `driver_backend_invocation_mismatch`;
- `driver_backend_runtime_path_mismatch`;
- `driver_backend_descriptor_token_mismatch`;
- `driver_backend_rank_device_mismatch`;
- `driver_backend_status_sink_mismatch`;
- `driver_backend_public_api_sourced_state`;
- `driver_backend_fabricated_pass_evidence`.

The focused red check failed first after adding only the new private-entry
test because the private backend scaffold/status owner symbols were missing.
The failure included missing
`runtime_dispatch_driver_backend_scaffold_status`,
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private driver backend request. It remains narrower than real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, H200 fused success,
public API expansion, examples, stable docs, or performance claims.

## Runtime Dispatch Driver Backend Request Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-map`.

Objective: map the future private driver backend request after PR #189
(`707cc81a818fdc00e4f592acb2f17538d1f6eb0a`). This docs/test dependency
slice defines ownership and validation boundaries only. It consumes the
backend scaffold/status input from PR #189 but does not implement real
UCCL-EP dispatch/combine work.

This map follows PR #189, which accepted only
`PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_status_name`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`.
The valid backend scaffold/status remains `unsupported`; it is not a backend
request, dispatch request, combine request, or pass-evidence source.

Boundary ownership:

- private backend request owner: the future private driver backend request
  owner accepts the PR #189 backend scaffold/status input only after the
  backend owner, invocation id, runtime path, descriptor token, rank/device
  map, world size, and runtime-owned status sink match;
- dispatch request placeholder: a private driver-owned placeholder for the
  future UCCL-EP dispatch request, with no transport call, payload transfer,
  kernel launch, scheduler transition, or pass evidence;
- combine request placeholder: a private driver-owned placeholder for the
  future UCCL-EP combine request, with no reduce/combine transport, payload
  release, kernel launch, scheduler transition, or pass evidence;
- descriptor token validation: the backend request must reuse the accepted
  backend scaffold/status descriptor token and must fail rather than create a
  token from hand-authored review data;
- rank/device validation: the backend request rank/device map must match the
  accepted backend scaffold/status rank, CUDA device, and world-size metadata;
- status sink ownership: the runtime-owned output sink remains the only
  review-facing status sink for request status and failure names;
- invalid public/provenance sources: example JSON, adapter-only provenance,
  public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, and hand-authored review artifacts cannot
  source backend request state or pass evidence.

Unsupported backend-request states:

- `driver_backend_request_pending`;
- `driver_backend_dispatch_request_placeholder`;
- `driver_backend_combine_request_placeholder`;
- `driver_backend_request_status_sink_unbound`;
- `driver_backend_request_map_unsupported_boundary`.

Failed backend-request states:

- `driver_backend_request_owner_mismatch`;
- `driver_backend_request_invocation_mismatch`;
- `driver_backend_request_runtime_path_mismatch`;
- `driver_backend_request_descriptor_token_mismatch`;
- `driver_backend_request_rank_device_mismatch`;
- `driver_backend_request_status_sink_mismatch`;
- `driver_backend_request_public_api_sourced_state`;
- `driver_backend_request_provenance_sourced_state`;
- `driver_backend_request_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
backend request scaffold/status. It must remain narrower than real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, H200 fused success,
public API expansion, examples, stable docs, or performance claims.

## Runtime Dispatch Driver Backend Request Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-scaffold-status`.

PR:
`#191` <https://github.com/uv-xiao/pto-cu/pull/191>.

Objective: implement the private backend request scaffold/status layer
selected by the PR #190 backend request map after
`4223edd9fa3c5e58b62eff1d7c27b1a54670766d`. The slice adds only
header-local private ABI plumbing and private-entry coverage for the future
runtime dispatch driver's backend request owner. It does not implement real
UCCL-EP dispatch/combine work.

Implementation evidence:

- `src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h` adds
  `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus`;
- the same header adds
  `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestStatus`;
- the same header adds
  `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_request_status_name`;
- the same header adds
  `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status`;
- the same header adds
  `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD`;
- `tests/ut/py/test_cuda_runtime_fusion_private_entry.py` adds
  `test_private_runtime_dispatch_driver_backend_request_scaffold_status_is_backend_owned`.

The valid prepared backend request scaffold/status remains `unsupported`,
`actual_fused_cross_gpu_execution` remains `0`, and no passed status or
transport/backend execution is reported. The private backend request scaffold
records backend request ownership, backend scaffold/status input, invocation
id, runtime path, status sink, descriptor token, and rank/device/world
metadata only.

Unsupported backend-request scaffold states:

- `driver_backend_request_pending`;
- `driver_backend_dispatch_request_placeholder`;
- `driver_backend_combine_request_placeholder`;
- `driver_backend_request_status_sink_unbound`;
- `driver_backend_request_map_unsupported_boundary`.

Failed backend-request scaffold states:

- `driver_backend_request_owner_mismatch`;
- `driver_backend_request_invocation_mismatch`;
- `driver_backend_request_runtime_path_mismatch`;
- `driver_backend_request_descriptor_token_mismatch`;
- `driver_backend_request_rank_device_mismatch`;
- `driver_backend_request_status_sink_mismatch`;
- `driver_backend_request_public_api_sourced_state`;
- `driver_backend_request_provenance_sourced_state`;
- `driver_backend_request_fabricated_pass_evidence`.

The focused red check failed first after adding only the new private-entry
test because the private backend request scaffold/status owner symbols were
missing. The failure included missing
`runtime_dispatch_driver_backend_request_scaffold_status`,
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_REQUEST_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status`.
The focused green check passed with `1 passed in 0.42s`, and the full
private-entry pytest passed with `19 passed in 5.11s`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private dispatch request placeholder. It remains narrower than real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, H200 fused success,
public API expansion, examples, stable docs, or performance claims.

## Runtime Dispatch Driver Backend Dispatch Request Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-map`.

Objective: map the future private dispatch request placeholder after PR #191
(`d4cbbfc130b356d90b649aa40f2c904d0fc8a081`). This docs/test dependency
slice defines ownership and validation boundaries only. It consumes the
backend request scaffold/status input from PR #191 but does not implement
real UCCL-EP dispatch/combine work.

This map follows PR #191, which accepted only
`PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendRequestStatus`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status`.
The valid backend request scaffold/status remains `unsupported`; it is not a
dispatch request, dispatch payload descriptor, dispatch output/status sink,
transport call, scheduler transition, or pass-evidence source.

Boundary ownership:

- private dispatch request placeholder owner: the future private dispatch
  request owner accepts the PR #191 backend request scaffold/status input only
  after backend request owner, invocation id, runtime path, descriptor token,
  rank/device map, world size, and runtime-owned dispatch output/status sink
  match;
- backend request scaffold/status input: the accepted PR #191 scaffold/status
  is the only private input to this map and remains unsupported until a later
  scaffold/status implementation owns dispatch request state;
- dispatch payload descriptor placeholder: a private driver-owned placeholder
  for future UCCL-EP dispatch payload descriptor vocabulary, with no payload
  transfer, descriptor allocation, transport call, kernel launch, scheduler
  transition, or pass evidence;
- dispatch output/status sink: the runtime-owned output/status sink remains
  the only review-facing sink for dispatch request status and failure names;
- descriptor token validation: the dispatch request placeholder must reuse
  the backend request scaffold/status descriptor token and must fail rather
  than create a token from hand-authored review data;
- rank/device validation: the dispatch request rank/device map must match the
  accepted backend request scaffold/status rank, CUDA device, and world-size
  metadata;
- invalid public/provenance sources: example JSON, adapter-only provenance,
  public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, and hand-authored review artifacts cannot
  source dispatch request state or pass evidence.

Unsupported dispatch-request states:

- `driver_backend_dispatch_request_pending`;
- `driver_backend_dispatch_payload_descriptor_placeholder`;
- `driver_backend_dispatch_output_status_sink_unbound`;
- `driver_backend_dispatch_request_map_unsupported_boundary`;
- `driver_backend_dispatch_payload_transfer_unimplemented`.

Failed dispatch-request states:

- `driver_backend_dispatch_request_owner_mismatch`;
- `driver_backend_dispatch_request_invocation_mismatch`;
- `driver_backend_dispatch_request_scaffold_mismatch`;
- `driver_backend_dispatch_request_descriptor_token_mismatch`;
- `driver_backend_dispatch_request_rank_device_mismatch`;
- `driver_backend_dispatch_request_status_sink_mismatch`;
- `driver_backend_dispatch_request_public_api_sourced_state`;
- `driver_backend_dispatch_request_provenance_sourced_state`;
- `driver_backend_dispatch_request_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
dispatch request scaffold/status only. It must remain narrower than real
UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, H200 fused
success, public API expansion, examples, stable docs, or performance claims.

## Runtime Dispatch Driver Backend Dispatch Request Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-scaffold-status`.

Objective: implement private dispatch request scaffold/status after PR #192
(`14aaedd8865ea7351cd30ee1a0dc46804b7d0f36`). This implementation slice
adds private ABI vocabulary only. It consumes the backend request
scaffold/status input from PR #191 and the dispatch request map from PR #192,
but does not implement real UCCL-EP dispatch/combine work.

Implementation evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_dispatch_request_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_dispatch_request_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_dispatch_request_scaffold_status_is_backend_owned`.

The valid prepared dispatch request scaffold/status remains `unsupported`;
`actual_fused_cross_gpu_execution` remains `0`; no passed status is reported;
and no payload transfer, transport/backend execution, scheduler pass, or H200
fused success is added. Malformed or stale private dispatch request owner
state produces a failed private result with dispatch-request vocabulary.
Focused red check failed first with `1 failed in 0.41s`; focused green check
passed with `1 passed in 0.41s`.

Unsupported dispatch-request states:

- `driver_backend_dispatch_request_pending`;
- `driver_backend_dispatch_payload_descriptor_placeholder`;
- `driver_backend_dispatch_output_status_sink_unbound`;
- `driver_backend_dispatch_request_map_unsupported_boundary`;
- `driver_backend_dispatch_payload_transfer_unimplemented`.

Failed dispatch-request states:

- `driver_backend_dispatch_request_owner_mismatch`;
- `driver_backend_dispatch_request_invocation_mismatch`;
- `driver_backend_dispatch_request_scaffold_mismatch`;
- `driver_backend_dispatch_request_descriptor_token_mismatch`;
- `driver_backend_dispatch_request_rank_device_mismatch`;
- `driver_backend_dispatch_request_status_sink_mismatch`;
- `driver_backend_dispatch_request_public_api_sourced_state`;
- `driver_backend_dispatch_request_provenance_sourced_state`;
- `driver_backend_dispatch_request_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine request placeholder. It remains narrower than real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, H200 fused success,
public API expansion, examples, stable docs, or performance claims.

## Runtime Dispatch Driver Backend Combine Request Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-map`.

Objective: define only the future private combine request placeholder after
PR #193 (`f969ea00c6858a6633ee53fd33bf77dd434097dc`). This docs/test
dependency map consumes the backend request scaffold/status input and the
dispatch request scaffold/status dependency, but it does not implement source
behavior, payload transfer, transport/backend execution, scheduler pass
evidence, or H200 fused success.

PR #193 is accepted only for private backend dispatch request scaffold/status
vocabulary and evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_dispatch_request_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_dispatch_request_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD`.

Boundary ownership:

- private combine request placeholder owner: the future private combine
  request owner accepts only the PR #193 dispatch request scaffold/status
  dependency after backend request owner, dispatch request owner, invocation
  id, runtime path, descriptor token, rank/device map, world size, and
  runtime-owned combine output/status sink match;
- backend request scaffold/status input: the earlier backend request
  scaffold/status remains an unsupported prerequisite and cannot source
  combine request state or pass evidence;
- dispatch request scaffold/status dependency: the PR #193 private dispatch
  request scaffold/status must be prepared and same-invocation before any
  future combine request placeholder can be mapped;
- combine payload descriptor placeholder: future placeholders only for
  UCCL-EP combine payload descriptor vocabulary, with no descriptor
  allocation, payload transfer, transport call, kernel launch, scheduler
  transition, or pass evidence;
- combine output/status sink: the runtime-owned output/status sink remains
  the only review-facing sink for combine request status and failure names;
- descriptor token validation: the combine request placeholder must reuse the
  dispatch request scaffold/status descriptor token and fail rather than
  create a token from hand-authored review data;
- rank/device validation: the combine request rank/device map must match the
  dispatch request scaffold/status rank, CUDA device, and world-size metadata;
- invalid public/provenance sources: example JSON, adapter-only provenance,
  public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, and hand-authored review artifacts cannot
  source combine request state or pass evidence.

Unsupported combine-request states:

- `driver_backend_combine_request_pending`;
- `driver_backend_combine_payload_descriptor_placeholder`;
- `driver_backend_combine_output_status_sink_unbound`;
- `driver_backend_combine_request_map_unsupported_boundary`;
- `driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-request states:

- `driver_backend_combine_request_owner_mismatch`;
- `driver_backend_combine_request_invocation_mismatch`;
- `driver_backend_combine_request_scaffold_mismatch`;
- `driver_backend_combine_request_descriptor_token_mismatch`;
- `driver_backend_combine_request_rank_device_mismatch`;
- `driver_backend_combine_request_status_sink_mismatch`;
- `driver_backend_combine_request_public_api_sourced_state`;
- `driver_backend_combine_request_provenance_sourced_state`;
- `driver_backend_combine_request_fabricated_pass_evidence`.

The map unsupported boundary and payload transfer unimplemented vocabulary
are future placeholders only.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, serving, vLLM, DeepSeek, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
combine request scaffold/status only. It must remain narrower than real
UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, H200 fused
success, public API expansion, examples, stable docs, serving, vLLM,
DeepSeek, or performance claims.

## Runtime Dispatch Driver Backend Combine Request Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-scaffold-status`.

Objective: implement private combine request scaffold/status after PR #194
(`562778f051ca87cf3f62d796860a8fd4c3476a32`). This implementation slice
adds private ABI vocabulary only. It consumes the backend request
scaffold/status input, the dispatch request scaffold/status dependency, and
the combine request map from PR #194, but does not implement real UCCL-EP
dispatch/combine work.

Implementation evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_request_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_request_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_combine_request_scaffold_status_is_backend_owned`.

The valid prepared combine request scaffold/status remains `unsupported`;
`actual_fused_cross_gpu_execution` remains `0`; no passed status is reported;
and no payload transfer, transport/backend execution, scheduler pass, or H200
fused success is added. Malformed or stale private combine request owner state
produces a failed private result with combine-request vocabulary. focused red
check failed first with `1 failed in 0.42s`; focused green check passed with
`1 passed in 0.42s`.

Unsupported combine-request states:

- `driver_backend_combine_request_pending`;
- `driver_backend_combine_payload_descriptor_placeholder`;
- `driver_backend_combine_output_status_sink_unbound`;
- `driver_backend_combine_request_map_unsupported_boundary`;
- `driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-request states:

- `driver_backend_combine_request_owner_mismatch`;
- `driver_backend_combine_request_invocation_mismatch`;
- `driver_backend_combine_request_scaffold_mismatch`;
- `driver_backend_combine_request_descriptor_token_mismatch`;
- `driver_backend_combine_request_rank_device_mismatch`;
- `driver_backend_combine_request_status_sink_mismatch`;
- `driver_backend_combine_request_public_api_sourced_state`;
- `driver_backend_combine_request_provenance_sourced_state`;
- `driver_backend_combine_request_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, serving, vLLM, DeepSeek, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine payload descriptor placeholder. It must remain narrower than
real UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, H200
fused success, public API expansion, examples, stable docs, serving, vLLM,
DeepSeek, or performance claims.

## Runtime Dispatch Driver Backend Combine Payload Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-map`.

Objective: define only the future private combine payload descriptor
placeholder after PR #195
(`e09b67a7a00f481f8c9dd4508d1adc9e88030d00`). This docs/test dependency map
consumes the backend request scaffold/status input, the dispatch request
scaffold/status dependency, and the combine request scaffold/status
dependency. It does not implement source behavior, descriptor allocation,
payload transfer, transport/backend execution, scheduler pass evidence, or
H200 fused success.

PR #195 is accepted only for private backend combine request scaffold/status
vocabulary and evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_request_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_request_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD`.

Boundary ownership:

- private combine payload descriptor placeholder owner: the future private
  combine payload owner accepts only same-invocation backend request,
  dispatch request, and combine request scaffold/status state after owner,
  invocation id, runtime path, descriptor token, rank/device map, world size,
  and runtime-owned combine payload output/status sink match;
- backend request scaffold/status input: the earlier backend request
  scaffold/status remains an unsupported prerequisite and cannot source
  combine payload state or pass evidence;
- dispatch request scaffold/status dependency: the private dispatch request
  scaffold/status dependency provides the descriptor-token and rank/device
  dependency that the future combine payload descriptor placeholder must
  reuse;
- combine request scaffold/status dependency: the PR #195 private combine
  request scaffold/status dependency must be prepared and same-invocation
  before any future combine payload descriptor placeholder can be mapped;
- combine payload output/status sink: the runtime-owned output/status sink
  remains the only review-facing sink for combine payload descriptor status
  and failure names;
- descriptor token validation: the combine payload descriptor placeholder must
  reuse the dispatch and combine request scaffold/status descriptor token and
  fail rather than create a token from hand-authored review data;
- rank/device validation: the combine payload rank/device map must match the
  dispatch and combine request scaffold/status rank, CUDA device, and
  world-size metadata;
- invalid public/provenance sources: example JSON, adapter-only provenance,
  public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, and hand-authored review artifacts cannot
  source combine payload descriptor state or pass evidence.

Unsupported combine-payload states:

- `driver_backend_combine_payload_pending`;
- `driver_backend_combine_payload_descriptor_placeholder`;
- `driver_backend_combine_payload_output_status_sink_unbound`;
- `driver_backend_combine_payload_map_unsupported_boundary`;
- `driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-payload states:

- `driver_backend_combine_payload_owner_mismatch`;
- `driver_backend_combine_payload_invocation_mismatch`;
- `driver_backend_combine_payload_request_scaffold_mismatch`;
- `driver_backend_combine_payload_descriptor_token_mismatch`;
- `driver_backend_combine_payload_rank_device_mismatch`;
- `driver_backend_combine_payload_status_sink_mismatch`;
- `driver_backend_combine_payload_public_api_sourced_state`;
- `driver_backend_combine_payload_provenance_sourced_state`;
- `driver_backend_combine_payload_fabricated_pass_evidence`.

The map unsupported boundary and payload transfer unimplemented vocabulary
are future placeholders only.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, serving, vLLM, DeepSeek, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
combine payload descriptor scaffold/status only. It must remain narrower than
real UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, H200
fused success, public API expansion, examples, stable docs, serving, vLLM,
DeepSeek, or performance claims.

## Runtime Dispatch Driver Backend Combine Payload Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-scaffold-status`.

Objective: implement private combine payload descriptor scaffold/status after
PR #196 (`af62d14d456b34fb1ef7fb2f9b4b6af7bc0bd4d1`). This slice consumes
the PR #196 private combine payload descriptor placeholder map and the PR #195
combine request scaffold/status dependency, but does not implement descriptor
allocation behavior, payload transfer, transport/backend execution, scheduler
pass evidence, or H200 fused success.

Accepted private ABI/status vocabulary:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_combine_payload_scaffold_status_is_backend_owned`.

The valid prepared combine payload descriptor scaffold/status remains
`unsupported`; `actual_fused_cross_gpu_execution` remains `0`. Malformed
private combine payload descriptor scaffold/status owner, invocation,
combine-request dependency, descriptor token, rank/device, sink, public
source, provenance source, or fabricated pass-evidence state fails the private
runtime-fusion entry instead of sourcing pass evidence.

Unsupported combine-payload states:

- `driver_backend_combine_payload_pending`;
- `driver_backend_combine_payload_descriptor_placeholder`;
- `driver_backend_combine_payload_output_status_sink_unbound`;
- `driver_backend_combine_payload_map_unsupported_boundary`;
- `driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-payload states:

- `driver_backend_combine_payload_owner_mismatch`;
- `driver_backend_combine_payload_invocation_mismatch`;
- `driver_backend_combine_payload_request_scaffold_mismatch`;
- `driver_backend_combine_payload_descriptor_token_mismatch`;
- `driver_backend_combine_payload_rank_device_mismatch`;
- `driver_backend_combine_payload_status_sink_mismatch`;
- `driver_backend_combine_payload_public_api_sourced_state`;
- `driver_backend_combine_payload_provenance_sourced_state`;
- `driver_backend_combine_payload_fabricated_pass_evidence`.

Focused TDD evidence: focused red check failed first with
`1 failed in 0.44s`; focused green check passed with `1 passed in 0.43s`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no descriptor allocation behavior;
- no payload transfer;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, serving, vLLM, DeepSeek, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine payload transfer boundary.

## Runtime Dispatch Driver Backend Combine Payload Transfer Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-map`.

Objective: map the future private combine payload transfer boundary after
PR #197 (`3337516c95fcd5f6129c515585d92e3f95f0c444`). This is a
dependency map only and must not implement source behavior. It accepts
PR #197 only for private backend combine payload descriptor scaffold/status
vocabulary and evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`.

The future private combine payload transfer boundary consumes backend request
scaffold/status input, the dispatch request scaffold/status dependency, the
combine request scaffold/status dependency, and the combine payload
descriptor scaffold/status dependency. The private combine payload transfer
boundary owner accepts only same-invocation private state after descriptor
token validation, rank/device validation, and combine payload transfer
output/status sink validation all match. Invalid public/provenance sources
remain forbidden inputs.

Unsupported combine-payload-transfer states:

- `driver_backend_combine_payload_transfer_pending`;
- `driver_backend_combine_payload_transfer_descriptor_placeholder`;
- `driver_backend_combine_payload_transfer_output_status_sink_unbound`;
- `driver_backend_combine_payload_transfer_map_unsupported_boundary`;
- `driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-payload-transfer states:

- `driver_backend_combine_payload_transfer_owner_mismatch`;
- `driver_backend_combine_payload_transfer_invocation_mismatch`;
- `driver_backend_combine_payload_transfer_payload_scaffold_mismatch`;
- `driver_backend_combine_payload_transfer_descriptor_token_mismatch`;
- `driver_backend_combine_payload_transfer_rank_device_mismatch`;
- `driver_backend_combine_payload_transfer_status_sink_mismatch`;
- `driver_backend_combine_payload_transfer_public_api_sourced_state`;
- `driver_backend_combine_payload_transfer_provenance_sourced_state`;
- `driver_backend_combine_payload_transfer_fabricated_pass_evidence`.

The map unsupported boundary and payload transfer unimplemented vocabulary
are future placeholders only.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no descriptor allocation behavior;
- no payload transfer implementation;
- no transport/backend execution;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, serving, vLLM, DeepSeek, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
combine payload transfer scaffold/status only. It must remain narrower than
real UCCL-EP dispatch/combine work, descriptor allocation behavior, payload
transfer implementation, transport/backend execution, scheduler/runtime pass
evidence, H200 fused success, public API expansion, examples, stable docs,
serving, vLLM, DeepSeek, or performance claims.

## Runtime Dispatch Driver Backend Combine Payload Transfer Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-scaffold-status`.

Objective: implement private combine payload transfer scaffold/status after
PR #198 (`227eae4c34a1182aab3548951380379da4582dc8`). This slice consumes
the PR #198 private combine payload transfer map and the accepted private
backend request, dispatch request, combine request, and combine payload
descriptor scaffold/status dependencies. It does not implement payload
transfer source behavior.

Accepted private ABI/status vocabulary:

- `PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status`;
- `test_private_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status_is_backend_owned`.

The current `PtoCudaRuntimeFusionFailure` mask already uses all 32 bits
through `1U << 31U`. This slice therefore does not add `1U << 32U` and does
not widen public ABI fields. Transfer scaffold/status failures reuse the
existing combine-payload scaffold aggregate failure bit,
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`,
while the transfer status enum and status-name function keep the private
boundary unambiguous.

The valid prepared combine payload transfer scaffold/status remains
`unsupported`; `actual_fused_cross_gpu_execution` remains `0`. Malformed
owner, invocation, payload scaffold dependency, descriptor token, rank/device,
status sink, public source, provenance source, or fabricated pass-evidence
state fails the private runtime-fusion entry instead of sourcing pass
evidence. The transfer scaffold/status also records same-invocation backend
request scaffold/status input, dispatch request scaffold/status dependency,
combine request scaffold/status dependency, and combine payload descriptor
scaffold/status dependency.

Unsupported combine-payload-transfer scaffold/status states:

- `driver_backend_combine_payload_transfer_pending`;
- `driver_backend_combine_payload_transfer_unimplemented`;
- `driver_backend_combine_payload_transfer_output_status_sink_unbound`;
- `driver_backend_combine_payload_transfer_map_unsupported_boundary`.

Failed combine-payload-transfer scaffold/status states:

- `driver_backend_combine_payload_transfer_owner_mismatch`;
- `driver_backend_combine_payload_transfer_invocation_mismatch`;
- `driver_backend_combine_payload_transfer_request_scaffold_mismatch`;
- `driver_backend_combine_payload_transfer_dispatch_request_scaffold_mismatch`;
- `driver_backend_combine_payload_transfer_combine_request_scaffold_mismatch`;
- `driver_backend_combine_payload_transfer_payload_scaffold_mismatch`;
- `driver_backend_combine_payload_transfer_descriptor_token_mismatch`;
- `driver_backend_combine_payload_transfer_rank_device_mismatch`;
- `driver_backend_combine_payload_transfer_status_sink_mismatch`;
- `driver_backend_combine_payload_transfer_public_api_sourced_state`;
- `driver_backend_combine_payload_transfer_provenance_sourced_state`;
- `driver_backend_combine_payload_transfer_fabricated_pass_evidence`.

Focused TDD evidence: focused red check failed first with
`1 failed in 0.46s`; focused green check passed with `1 passed in 0.45s`.
Final verification passed: the final focused green rerun passed with `1 passed`; `git diff --check`
passed; targeted markdownlint over the five NVIDIA in-progress docs reported
`0 error(s)`; the NVIDIA review guard passed;
`test_cuda_runtime_fusion_private_entry.py` passed with `23 passed`; and
`test_nvidia_review_artifacts.py` passed with `85 passed`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no descriptor allocation behavior change;
- no payload transfer implementation;
- no transport/backend execution;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public API expansion;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples, stable docs, serving, vLLM, DeepSeek, or performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine payload transfer completion boundary. It must remain a map,
not a payload transfer implementation.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-map`.

Objective: map the future private combine payload transfer completion boundary
after PR #199 (`41c9c894ad511534d943180bccb10aab8fba3f7b`). This is a
dependency map only, not source behavior.

Accepted PR #199 evidence is limited to private combine payload transfer
scaffold/status vocabulary:
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status`,
and
`test_private_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status_is_backend_owned`.

Failure-bit design note: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. Future
transfer completion scaffold/status failures must reuse
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
as the existing combine-payload scaffold aggregate failure bit.

The future private combine payload transfer completion boundary consumes
backend request scaffold/status input, the dispatch request scaffold/status
dependency, the combine request scaffold/status dependency, the combine
payload descriptor scaffold/status dependency, and the combine payload
transfer scaffold/status dependency.

Future validation must remain private to the same invocation. Required
ownership and validation checks are same invocation, transfer owner, transfer
status dependency, descriptor token, rank/device, status sink, completion
sink, no public/provenance sourced state, and no fabricated pass evidence.

Completion status vocabulary:
`driver_backend_combine_payload_transfer_completion_pending`,
`driver_backend_combine_payload_transfer_completion_unimplemented`,
`driver_backend_combine_payload_transfer_completion_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_transfer_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_completion_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_fabricated_pass_evidence`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no descriptor allocation behavior change;
- no payload transfer implementation;
- no completion implementation;
- no transport/backend execution;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public API expansion;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples/stable docs/serving/vLLM/DeepSeek/performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
completion scaffold/status only.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-scaffold-status`.

Objective: implement private combine payload transfer completion
scaffold/status after PR #200
(`47136ab0b1cff42b7ad3448809a3b9e5bf44db43`). This is a private
completion scaffold/status only, not payload transfer or completion
execution.

Accepted PR #200 evidence is limited to the private combine payload transfer
completion boundary map and required vocabulary:
`driver_backend_combine_payload_transfer_completion_pending`,
`driver_backend_combine_payload_transfer_completion_unimplemented`,
`driver_backend_combine_payload_transfer_completion_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_transfer_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_completion_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_fabricated_pass_evidence`.

Accepted private ABI/status vocabulary for this slice:
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status`,
and
`test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status_is_backend_owned`.

The prepared completion scaffold/status consumes backend request
scaffold/status input, the dispatch request scaffold/status dependency, the
combine request scaffold/status dependency, the combine payload descriptor
scaffold/status dependency, and the combine payload transfer scaffold/status
dependency.

Coordinator-owned validation fails closed for owner, invocation, transfer
scaffold dependency, descriptor token, rank/device, status sink, completion
sink, public/provenance sourced state, and fabricated pass evidence. The
valid prepared completion scaffold/status remains unsupported, and
`actual_fused_cross_gpu_execution` remains `0`.

Failure-bit design note: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. Completion
scaffold/status validation reuses
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
as the existing combine-payload scaffold aggregate failure bit while keeping
completion status enum and status-name vocabulary unambiguous.

Red/green evidence:

- focused red check failed first after adding only
  `test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status_is_backend_owned`;
  the exact focused command failed with `1 failed in 0.47s` because the
  coordinator member, version constant, completion status constants,
  status-name function, and prepare function were missing;
- focused green check passed with `1 passed in 0.44s` after adding the
  private completion scaffold/status implementation.

Final verification passed: focused green passed;
`git diff --check` passed with no output; targeted `markdownlint-cli2`
reported `Summary: 0 error(s)`; the NVIDIA review guard reported
`nvidia review guard passed`; the full private runtime-fusion pytest reported
`24 passed`; and the NVIDIA review-artifact pytest reported `87 passed`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no descriptor allocation behavior change;
- no payload transfer implementation;
- no completion implementation;
- no transport/backend execution;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public API expansion;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples/stable docs/serving/vLLM/DeepSeek/performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-map`.
This is selected exactly one next PR-sized docs/test map slice for the future
private combine payload transfer completion handoff boundary. It is not a
payload transfer implementation or completion implementation.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-map`.

Objective: map the future private combine payload transfer completion handoff
boundary after PR #201 (`47e7bd1e`). This is a dependency map only, not
source behavior.

PR #201 is accepted only for private combine payload transfer completion
scaffold/status vocabulary and evidence:
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status`,
and
`test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status_is_backend_owned`.

Failure-bit design note: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. Completion
scaffold/status failures reuse
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
as the existing combine-payload scaffold aggregate failure bit.

The future private combine payload transfer completion handoff boundary
consumes backend request scaffold/status input, the dispatch request
scaffold/status dependency, the combine request scaffold/status dependency,
the combine payload descriptor scaffold/status dependency, the combine
payload transfer scaffold/status dependency, and the combine payload transfer
completion scaffold/status dependency.

Future handoff validation must remain private to the same invocation.
Required ownership and validation checks are same invocation, handoff owner,
completion status dependency, transfer scaffold dependency, descriptor token,
rank/device, status sink, handoff sink, no public/provenance sourced state,
and no fabricated pass evidence.

Completion handoff status vocabulary:
`driver_backend_combine_payload_transfer_completion_handoff_pending`,
`driver_backend_combine_payload_transfer_completion_handoff_unimplemented`,
`driver_backend_combine_payload_transfer_completion_handoff_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_handoff_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_completion_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_transfer_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_handoff_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_handoff_fabricated_pass_evidence`.

Focused red check failed first with `1 failed in 0.95s`; the missing
review-artifact section was
`Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Map Slice`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no descriptor allocation behavior change;
- no payload transfer implementation;
- no completion implementation;
- no handoff implementation;
- no transport/backend execution;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public API expansion;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples/stable docs/serving/vLLM/DeepSeek/performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
handoff scaffold/status only.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-scaffold-status`.

Objective: implement private combine payload transfer completion handoff
scaffold/status after merged PR #202 (`925eed3e`). This is private
handoff scaffold/status only, not real payload transfer, completion, or
handoff execution.

PR #202 is accepted as the docs/test map for the future private combine
payload transfer completion handoff boundary. This slice promotes that map
into private ABI/status vocabulary and coordinator-owned validation.

New private vocabulary and evidence:
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status`,
and
`test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status_is_backend_owned`.

The handoff scaffold/status depends on backend request scaffold/status input,
the dispatch request scaffold/status dependency, the combine request
scaffold/status dependency, the combine payload descriptor scaffold/status
dependency, the combine payload transfer scaffold/status dependency, and the
combine payload transfer completion scaffold/status dependency.

Valid prepared handoff scaffold/status remains `unsupported`, keeps
`actual_fused_cross_gpu_execution == 0`, and binds status sink and handoff
sink to coordinator-owned output state. Validation fails closed for owner,
invocation, completion scaffold dependency, transfer scaffold dependency,
descriptor token, rank/device, status sink/handoff sink,
public/provenance sourced state, and fabricated pass evidence.

Handoff status vocabulary:
`driver_backend_combine_payload_transfer_completion_handoff_pending`,
`driver_backend_combine_payload_transfer_completion_handoff_unimplemented`,
`driver_backend_combine_payload_transfer_completion_handoff_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_handoff_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_completion_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_transfer_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_handoff_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_handoff_fabricated_pass_evidence`.

Failure-bit design note: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. Completion
handoff scaffold/status validation reuses
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
as the existing combine-payload scaffold aggregate failure bit while keeping
handoff status enum and status names unambiguous.

Red/green evidence: focused red check failed first with
`1 failed in 0.51s` because the handoff scaffold/status API was missing;
focused green check passed with `1 passed in 0.44s` after adding the private
handoff scaffold/status implementation.

Final verification passed before PR creation: focused green passed;
`git diff --check` passed with no output; targeted `markdownlint-cli2`
reported `Summary: 0 error(s)`; the NVIDIA review guard reported
`nvidia review guard passed`; the full private runtime-fusion pytest reported
`25 passed`; and the NVIDIA review-artifact pytest reported `89 passed`.

Required non-claims:

- no real UCCL-EP dispatch/combine work;
- no descriptor allocation behavior change;
- no payload transfer implementation;
- no completion implementation;
- no handoff implementation;
- no transport/backend execution;
- no scheduler/runtime pass evidence;
- no fresh H200 fused success;
- no public API expansion;
- no public `TaskArgs`;
- no public `CallConfig`;
- no common runtime C API;
- no UCCL host-runtime ABI;
- no examples/stable docs/serving/vLLM/DeepSeek/performance claims;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-map`.
This is selected exactly one next PR-sized docs/test map slice for the future
private completion handoff result boundary. It is not a real payload
transfer, completion, handoff, transport, or backend implementation.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Result Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-map`.

Objective: create the docs/test mapping only slice for the future private
completion handoff result boundary after merged PR #203 (`15a66e7b`). This
is not source behavior and it makes only review vocabulary visible.

PR #203 is accepted only for private combine payload transfer completion
handoff scaffold/status vocabulary and coordinator-owned validation. This
map depends on backend request scaffold/status input, the dispatch request
scaffold/status dependency, the combine request scaffold/status dependency,
the combine payload descriptor scaffold/status dependency, the combine
payload transfer scaffold/status dependency, the combine payload transfer
completion scaffold/status dependency, and the combine payload transfer
completion handoff scaffold/status dependency.

The future private completion handoff result boundary remains private to the
same invocation. Likely validation concepts are review vocabulary only, not
implementation claims: result owner, same invocation, backend request
dependency, dispatch request dependency, combine request dependency, combine
payload descriptor dependency, transfer scaffold/status dependency,
completion scaffold/status dependency, handoff scaffold/status dependency,
descriptor token, rank/device, status sink, result sink, no
public/provenance sourced state, and no fabricated pass evidence.

Completion handoff result status vocabulary:
`driver_backend_combine_payload_transfer_completion_handoff_result_pending`,
`driver_backend_combine_payload_transfer_completion_handoff_result_unimplemented`,
`driver_backend_combine_payload_transfer_completion_handoff_result_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_handoff_result_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_backend_request_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_dispatch_request_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_combine_request_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_descriptor_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transfer_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_completion_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_handoff_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_handoff_result_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_handoff_result_fabricated_pass_evidence`.

This slice records not real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer, no completion, no handoff
implementation, no transport/backend execution, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public API expansion, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples/stable docs/serving/vLLM/DeepSeek/performance claims.
It does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`
and does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
completion handoff result scaffold/status only.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Result Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-scaffold-status`.

Objective: implement private completion handoff result scaffold/status after
merged PR #204 (`f425090f`). This is private completion handoff result
scaffold/status only, not a real result transport/backend implementation.

PR #204 is accepted as the docs/test map for the private completion handoff
result boundary. This slice promotes that map into private ABI/status
vocabulary and coordinator-owned validation.

New private vocabulary and evidence:
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_RESULT_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffResultScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffResultStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status`,
and
`test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_result_scaffold_status_is_backend_owned`.

The result scaffold/status depends on backend request scaffold/status input,
the dispatch request scaffold/status dependency, the combine request
scaffold/status dependency, the combine payload descriptor scaffold/status
dependency, the combine payload transfer scaffold/status dependency, the
combine payload transfer completion scaffold/status dependency, and the
combine payload transfer completion handoff scaffold/status dependency.

valid prepared result scaffold/status remains `unsupported`, keeps
`actual_fused_cross_gpu_execution == 0`, and binds status sink and result
sink to coordinator-owned output state. Validation fails closed for result
owner, same invocation, handoff scaffold dependency, completion scaffold
dependency, transfer scaffold dependency, descriptor token, rank/device,
status sink/result sink, public/provenance sourced state, and fabricated
pass evidence. There is no actual fused execution claim.

Result status vocabulary:
`driver_backend_combine_payload_transfer_completion_handoff_result_pending`,
`driver_backend_combine_payload_transfer_completion_handoff_result_unimplemented`,
`driver_backend_combine_payload_transfer_completion_handoff_result_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_handoff_result_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_handoff_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_completion_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transfer_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_handoff_result_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_handoff_result_fabricated_pass_evidence`.

Failure-bit design note: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. Result
scaffold/status validation reuses
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
as the existing combine-payload scaffold aggregate failure bit while keeping
result status enum and status names unambiguous.

Red/green evidence: focused red check failed first with
`1 failed in 0.52s` because the result scaffold/status API was missing;
focused green check passed with `1 passed in 0.46s` after adding the private
result scaffold/status implementation. Focused review-artifact red failed
with `1 failed in 0.93s` before the status docs were synchronized; focused
review-artifact green passed with `1 passed in 0.06s`.

Final verification passed before PR creation: `git diff --check` passed with
no output; targeted `markdownlint-cli2` over the five NVIDIA in-progress
docs reported `Summary: 0 error(s)`; the NVIDIA review guard reported
`nvidia review guard passed`; the full private runtime-fusion pytest reported
`26 passed`; and the NVIDIA review-artifact pytest reported
`91 passed`.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no handoff implementation, no result transport
implementation, no transport/backend execution, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public API expansion, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples/stable docs/serving/vLLM/DeepSeek/performance claims.
It does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`
and does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-map`.
This is selected exactly one next PR-sized docs/test map slice for the future
private result transport boundary. It is not result transport implementation,
backend execution, or fused execution evidence.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Result Transport Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-map`.

Objective: create the docs/test mapping only slice for the future private
result transport boundary after merged PR #205 (`d80ccd23`). This is a
branch/slice review map only, not source behavior or transport execution.

PR #205 is accepted as private completion handoff result scaffold/status.
This map depends on backend request scaffold/status input, dispatch request
scaffold/status dependency, combine request scaffold/status dependency,
combine payload descriptor scaffold/status dependency, combine payload
transfer scaffold/status dependency, combine payload transfer completion
scaffold/status dependency, combine payload transfer completion handoff
scaffold/status dependency, completion handoff result scaffold/status
dependency, and handoff result scaffold/status dependency.

The future private result transport boundary remains private to the same
invocation. Likely validation concepts are review vocabulary only, not
implementation claims: transport owner, same invocation, descriptor token,
rank/device/world size, status sink, result sink, transport sink/handle, no
public/provenance sourced state, and no fabricated pass evidence.

Result transport status vocabulary:
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_pending`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_unimplemented`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_result_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_handle_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_handoff_result_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_rank_device_world_size_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_result_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_handle_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_fabricated_pass_evidence`.

Failure-bit constraint: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U`. Future implementation must reuse the
existing combine-payload aggregate failure bit or a documented existing bit,
not widen public ABI.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no handoff implementation, no result transport
implementation, no transport/backend execution, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public API expansion, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples/stable docs/serving/vLLM/DeepSeek/performance claims.
It does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`
and does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
result transport scaffold/status only.
