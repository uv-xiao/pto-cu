# Distributed Communication Selection

This note records the first Ray/NCCL/UCCL selection for the NVIDIA backend
restart. It is a planning boundary, not performance evidence.

## Role Summary

- Ray: process and serving orchestration. Keep as an orchestration option
  outside the compiled CUDA scheduler and worker path.
- NCCL: baseline collectives. Use as the first baseline for all-reduce,
  all-gather, reduce-scatter, broadcast, and send/receive comparisons on
  NVIDIA GPUs.
- UCCL: EP/P2P research path. Keep as an opt-in experimental adapter
  direction. Do not include UCCL host-runtime dispatch in the NCCL
  worker-control slice.

## Implications

- The simpler NVIDIA platform should expose communication through a narrow
  runtime boundary. The persistent-device megakernel may orchestrate device
  work, but it should not directly depend on Ray.
- NCCL is the default measurement baseline because it is the standard NVIDIA
  GPU collective library and covers both single-node and multi-node
  collectives.
- UCCL remains an opt-in research path for expert-parallel and P2P work. It
  can be composed with existing gates as Python-side adapter evidence, but it
  is not a CUDA host-runtime dispatch claim.
- Ray can matter later for `pto-serving`, vLLM integration, and cluster
  lifecycle, but the 2 H200 first milestone should prove local communication
  behavior before adding a Ray dependency.

## Evaluation Order

1. Use the established 2 H200 NCCL baseline in
   `docs/in_progress/nvidia_backend/nccl_two_h200_baseline.md` as the
   compatibility floor for all-reduce, reduce-scatter, all-gather, and
   point-to-point send/receive.
2. Implement the PTO-side communication boundary with mock and local
   single-host tests first; avoid baking NCCL types into public Python APIs.
3. Land descriptor-backed CUDA host-runtime NCCL communicator setup and
   worker-control operation dispatch.
4. Compare UCCL P2P or expert-parallel paths as separate opt-in slices after
   the NCCL worker-control baseline is accepted.
5. Evaluate Ray only after serving needs multi-process orchestration,
   autoscaling, placement, or request routing.

## Current Evidence

- NCCL baseline on 2 H200 exists:
  `docs/in_progress/nvidia_backend/nccl_two_h200_baseline.md`.
- PTO-side boundary constraints are recorded in
  `docs/in_progress/nvidia_backend/communication_runtime_boundary.md`.
- The baseline passed on `NVIDIA H200 NVL` devices `0,1` for `all_reduce`,
  `reduce_scatter`, `all_gather`, and `send_recv` with `world_size: 2`.
- The NCCL baseline acquires local-rank process-group state through
  `simpler_setup.cuda_comm.CudaCommRuntimeRegistry`.
- The NCCL baseline derives group capability and per-rank launch metadata from
  a host-side `CudaCommHostPlan`, preserving Worker-style `device_ids` without
  parsing transport-specific state.
- L3 CUDA `Worker` instances own the same host-plan metadata privately,
  keeping communication mapping at the worker fork boundary without exposing
  NCCL objects in task arguments.
- Each CUDA chip child receives its own private `CudaCommLaunchPlan` on the
  Python `ChipWorker` wrapper before task dispatch.
- The launch plan derives a compact `CudaCommDeviceDescriptor` so C++ host
  runtime work can consume fixed-size descriptor bytes instead of Python
  objects.
- The CUDA host runtime has a C++ ABI header and optional
  `configure_cuda_comm_descriptor()` entry that parses and stores those
  descriptor bytes.
- The CUDA chip-child binding path passes descriptor bytes into that host
  runtime entry point before the task loop starts.
- The CUDA host runtime has descriptor-backed communicator setup: `comm_init`
  dynamically loads `libnccl`, exchanges an `ncclUniqueId` through the
  root-info file, and owns `ncclComm_t` teardown through `comm_destroy`.
- CUDA host-runtime operation dispatch covers `comm_all_reduce_f32`,
  `comm_reduce_scatter_f32`, `comm_all_gather_f32`, and `comm_send_recv_f32`
  through the descriptor-backed NCCL comm handle.
- `ChipWorker` exposes those entries as internal device-pointer methods so
  forked chip children can drive the operation path without exposing NCCL
  transport objects in task arguments.
- The hierarchical worker control plane has `CTRL_COMM_OP` /
  `control_comm_op` transport for those internal `ChipWorker` methods, with
  request payloads staged in POSIX shared memory and serialized through the
  existing per-chip mailbox.
- Worker-control H200 evidence exists in
  `docs/in_progress/nvidia_backend/nccl_worker_control_h200.md`: on
  `NVIDIA H200 NVL` devices `6,7`, `examples/cuda/nccl_worker_control_ops.py`
  passed `all_reduce`, `reduce_scatter`, `all_gather`, and `send_recv`
  through `CTRL_COMM_OP` with `max_abs_error: 0.0`.
- UCCL-EP adapter handoff evidence exists in
  `docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`:
  on `NVIDIA H200 NVL` devices `6,7`, the opt-in
  `--with-uccl-ep-handoff` path passed by composing the existing two-device
  persistent MoE aggregate with the Python-side UCCL-EP dispatch/combine
  adapter on the same device ids.
- The reduced fused cross-GPU expert-parallel MoE boundary now has an explicit
  `--with-uccl-ep-fused-boundary` result shape. It records a structured
  unsupported boundary, not fused evidence, until
  `persistent_device_uccl_ep_runtime_fusion` exists and can prove actual fused
  cross-GPU expert-parallel MoE execution.
  It is a structured unsupported boundary.
- The UCCL-EP handoff and fused-boundary JSON now record
  `payload_provenance` from participating components only. Adapter provenance
  includes the UCCL capability id, descriptor dimensions, metadata shapes,
  rank results, and rank/device mapping. Persistent-device graph provenance
  includes the graph descriptor id, device ids, rank/device mapping, source
  digests, and bridge digest.
- No runtime component creates or transfers shared payload ownership in the
  current result. The JSON explicitly reports no shared ownership token and an
  empty lifetime transition log, while
  `persistent_device_uccl_ep_runtime_fusion.status` remains `unsupported`.
- This design/dependency slice selects
  `persistent_device_uccl_ep_runtime_fusion` as the next UCCL-EP dependency
  before implementation. The boundary must preserve the existing runtime
  separation: UCCL-EP remains opt-in and internal, rank/device mapping derives
  from Worker-local device ordering, and dispatch/combine payload ownership is
  recorded by the runtime result rather than exposed in public task APIs.
- A later implementation PR may keep the current H200 command shape and
  convert the fused-boundary result from `unsupported` to `passed` only after
  the persistent-device graph and UCCL-EP runtime share payload descriptors,
  payload ownership, rank/device mapping, status fields, and failure modes.
- The readiness map keeps those shared payload descriptors behind the
  `ChipWorker` to CUDA runtime boundary. A private
  persistent-device/UCCL-EP runtime fusion coordinator owns the shared
  descriptor, issues the ownership token, validates the payload lifetime
  transition log, and records descriptor, rank/device, payload lifetime,
  transport, scheduler, validation, and unsupported-boundary failures.
- PR #147 remains accepted provenance-only input evidence. It supplies adapter
  and graph payload fields, but it has no runtime-owned shared descriptor, no
  shared ownership token, and an empty lifetime transition log; therefore
  `persistent_device_uccl_ep_runtime_fusion.status` remains `unsupported` and
  `actual_fused_cross_gpu_execution` remains `false`.
- The first implementation attempt on
  `nvidia-uccl-ep-runtime-fusion-impl-runtime-owned-descriptor` found no real
  runtime-owned UCCL-EP fusion coordinator behind the CUDA runtime /
  `ChipWorker` boundary. It therefore adds local guards only:
  `_validate_runtime_fusion_evidence` rejects fabricated or incomplete pass
  evidence, records missing ownership in `failure_fields`, and keeps the
  normal fused-boundary result `unsupported`. Pass-like handoff metadata is
  rejected with `failure_fields.fabricated_or_untrusted_pass_evidence`.
- A later implementation branch should keep the same narrow scope but first
  add the missing runtime-owned coordinator behind the CUDA runtime /
  `ChipWorker` boundary. Only that lower-level boundary can create the shared
  descriptor, ownership token, lifetime log, and fresh H200 fused-boundary
  result. It must not add RDMA, multi-node transport, UCCL host-runtime ABI
  expansion, serving, vLLM, or DeepSeek claims.
- The coordinator-boundary map keeps the runtime owner concrete:
  `persistent_device_uccl_ep_runtime_fusion` is created inside the CUDA
  persistent-device runtime run context for one `ChipWorker::run` call. The
  descriptor allocation site, ownership token issuer, and lifetime state
  machine live there, not in example-side handoff metadata.
- The reviewable entry point is `WorkerThread` chip dispatch ->
  `ChipWorker::run` -> CUDA host-runtime callable entry ->
  persistent-device runtime run context -> runtime-fusion coordinator.
  Future pass evidence must record the coordinator-issued token and the
  `allocated` -> `dispatch_ready` -> `dispatch_in_flight` ->
  `combine_ready` -> `combine_in_flight` -> `complete` -> `released`
  transition log.
- The private entry contract names
  `persistent_device_uccl_ep_runtime_fusion_entry` as the CUDA
  persistent-device host-callable path reached after `ChipWorker::run`
  assembles `ChipStorageTaskArgs`. It does not add public `TaskArgs`,
  public `CallConfig`, or UCCL host-runtime ABI fields.
- `ChipWorker::run` requests coordinator construction with existing private
  runtime inputs: callable id, chip-local rank/device map, persistent graph
  descriptor handle, UCCL-EP capability metadata, descriptor allocation
  policy, validation policy, and a runtime-owned output sink.
- The entry result must return coordinator status, descriptor allocation
  provenance, coordinator-issued ownership token, ordered state transitions,
  rank/device map, validation summary, and explicit failure fields to the
  host/runtime status artifact.
- Example-side JSON, adapter-only provenance, handoff metadata, public
  `TaskArgs`, and public `CallConfig` remain forbidden pass-evidence paths.
  If they supply pass-like fields, the result must fail with
  `fabricated_or_untrusted_pass_evidence`.
- The private unsupported entry scaffold adds
  `src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h` and a CUDA
  host-runtime hook in
  `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp`. It constructs a
  private `PtoCudaRuntimeFusionRequest` from the callable id, persistent DAG
  graph descriptor, private CUDA rank/device descriptor when configured, and
  runtime-owned output sink. Missing coordinator, descriptor allocator,
  UCCL-EP runtime path, validation policy, UCCL-EP capability metadata, or
  `ChipStorageTaskArgs` keeps the result `unsupported` with explicit failure
  bits. Forbidden pass-evidence paths still fail as
  `fabricated_or_untrusted_pass_evidence`.
- The current accepted evidence boundary is unchanged: PR #147 remains
  provenance-only unsupported-boundary evidence, PR #150 remains guard-only
  blocked implementation evidence, PR #151 remains a post-PR150 status
  refresh, PR #152 remains a coordinator-boundary map only, and PR #153
  remains a private entry-contract only. PR #155 remains a private
  unsupported runtime scaffold only
  (`nvidia-uccl-ep-runtime-fusion-private-entry-unsupported`). None of those
  PRs proves actual fused cross-GPU expert-parallel MoE execution.
- PR #157
  (`nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request`) is closed
  invalid. It assigned a `PtoCudaPersistentDagArgs *` persistent DAG run
  pointer to `PtoCudaRuntimeFusionRequest::chip_storage_task_args` and labeled
  it `sizeof(ChipStorageTaskArgs)`, but that is not a real
  `ChipStorageTaskArgs *` from `ChipWorker::run`.
- The `nvidia-uccl-ep-runtime-fusion-private-request-envelope` dependency
  narrows that handoff. Its private envelope can carry runtime-specific task
  args separately from a typed `ChipStorageTaskArgs` pointer without expanding
  public `TaskArgs`, public `CallConfig`, common runtime C API, or UCCL
  host-runtime ABI fields. `ChipWorker::run` cannot provide the required
  `PtoCudaPersistentDagArgs *`, so it explicitly rejects the private-envelope
  path instead of fabricating runtime args. It keeps the fused-boundary result
  `unsupported` unless the runtime coordinator emits real descriptor
  ownership, ownership-token, lifetime-transition, rank/device, validation,
  and failure-field evidence.
- PR #161 selected
  `nvidia-uccl-ep-runtime-fusion-runtime-args-handoff-map` as the next
  dependency slice after the private request envelope. PR #162 accepted that
  handoff map as docs/test dependency evidence only. It keeps
  `ChipWorker::run` as the owner of the real `ChipStorageTaskArgs *` and the
  CUDA persistent DAG host-runtime path as the owner of the real
  `PtoCudaPersistentDagArgs *`.
- The private association point is inside the CUDA host runtime, after the
  persistent DAG callable is resolved for the same `ChipWorker::run`
  invocation. `PtoCudaPrivateRunArgsEnvelope` associates `runtime_task_args`
  with `chip_storage_task_args` only when both pointers are real,
  same-invocation inputs with matching sizes and callable type.
- Public `TaskArgs`, public `CallConfig`, the common runtime C API,
  UCCL host-runtime ABI fields, example JSON, adapter provenance, and handoff
  metadata remain forbidden ways to provide the runtime args association.
  Null, stale, wrong-size, wrong-callable, or cross-invocation envelopes must
  be failed or unsupported states, not pass evidence.
- PR #164's accepted implementation slice
  `nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff` adds only the
  private CUDA persistent DAG host-runtime association for real
  same-invocation `ChipStorageTaskArgs *` and `PtoCudaPersistentDagArgs *`
  inputs, with local coverage for null pointers, wrong sizes,
  mismatched-callable cases, stale envelopes, cross-invocation envelopes, and
  forbidden public/API evidence paths. It does not add a runtime-fusion
  coordinator, descriptor allocator, UCCL-EP runtime path, validation policy,
  UCCL-EP capability metadata, pass evidence, or fused-success claim.
- PR #165 accepted only the post-PR164 docs/test status refresh and selected
  the capability metadata map.
- PR #166 accepted only the private UCCL-EP capability metadata dependency
  map. It did not implement a runtime-fusion coordinator, descriptor
  allocator, UCCL-EP runtime path, validation policy, CUDA runtime behavior,
  pass evidence, or H200 fused-success evidence.
- PR #167 accepted only the post-PR166 docs/test status refresh and selected
  the validation policy map.
- PR #168 accepted only the private validation policy dependency map. It did
  not implement CUDA runtime behavior, descriptor allocation policy, UCCL-EP
  runtime dispatch, a coordinator, pass evidence, or H200 fused-success
  evidence.
- The next selected slice is
  `nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map`, a
  docs/test dependency map for the private descriptor allocation policy that a
  later coordinator request will need after validation policy exists. It must
  not implement runtime behavior, widen public `TaskArgs` / `CallConfig`,
  widen the common runtime C API or UCCL host-runtime ABI, implement UCCL-EP
  runtime dispatch, construct a coordinator, or claim H200 fused success.
- PR #170 accepted only the private descriptor allocation policy dependency
  map. It did not implement CUDA runtime behavior, descriptor allocation,
  UCCL-EP runtime dispatch, a coordinator, pass evidence, or H200
  fused-success evidence.
- PR #172 accepted only the private UCCL-EP runtime path dependency map. It
  did not implement CUDA runtime behavior, UCCL-EP runtime dispatch, a
  coordinator, descriptor allocation, pass evidence, or H200 fused-success
  evidence.
- PR #174 accepted only the private UCCL-EP runtime path scaffold:
  `PtoCudaUcclEpRuntimePath`, `PtoCudaUcclEpRuntimeDescriptorView`, private
  descriptor-view validation, and invocation-id propagation through private
  CUDA runtime-fusion request state. It did not implement the coordinator,
  descriptor allocation, UCCL-EP runtime dispatch, pass evidence, fresh H200
  fused-success evidence, public runtime API fields, serving, vLLM, DeepSeek,
  or performance evidence.
- PR #176 accepted only the private UCCL-EP runtime-fusion descriptor
  allocation scaffold: `PtoCudaUcclEpDescriptorHostControl`,
  `PtoCudaUcclEpDeviceDescriptorBuffer`,
  `PtoCudaUcclEpDescriptorAllocation`, and
  `pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors`. It did not
  go beyond the private host-control record and device-visible
  dispatch/combine descriptor buffer mechanics, and it did not implement
  coordinator construction, UCCL-EP runtime dispatch, pass evidence, fresh
  H200 fused-success evidence, public `TaskArgs`, public `CallConfig`, common
  runtime C API fields, UCCL host-runtime ABI fields, examples, stable docs,
  serving, vLLM, DeepSeek, or performance evidence.
  This is the accepted private descriptor allocation scaffold, not fused
  execution evidence.
- The branch `nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status`
  implements a narrow private runtime-fusion coordinator scaffold/status
  slice. It wires private coordinator state needed to own the PR #176
  descriptor allocation and PR #174 runtime path, but it remains narrower than
  UCCL-EP runtime dispatch and pass evidence. It cannot claim fused success
  until real UCCL-EP runtime dispatch and fresh H200 fused-boundary evidence
  exist.

## Capability Metadata Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-capability-metadata-map`.

PR #165 selected this dependency slice after PR #164 accepted only the
private host-runtime pointer association. This branch maps private UCCL-EP
capability metadata for the later
`persistent_device_uccl_ep_runtime_fusion_entry` coordinator request without
implementing CUDA runtime behavior.

The minimum private metadata fields are capability id, world size,
rank-to-device map, descriptor vocabulary, transport mode, adapter provenance
handles, and setup/validation failure ownership. They stay private to the
CUDA persistent-device runtime path and chip-child private metadata.

The PR #164 association between real same-invocation `ChipStorageTaskArgs *`
and `PtoCudaPersistentDagArgs *` is preserved. Capability metadata may only
describe the private UCCL-EP capability that the later coordinator consumes;
it cannot stand in for either same-invocation pointer.

The cases missing, stale, mismatched-rank, mismatched-world-size, or
public/API-sourced capability metadata remain unsupported or failed. Public
`TaskArgs`, public `CallConfig`, common runtime C API, UCCL host-runtime ABI,
example JSON, adapter provenance, and handoff metadata remain forbidden
pass-evidence paths.

This slice makes no CUDA runtime behavior change. It has no runtime-fusion
coordinator implementation, no descriptor allocator implementation, no
UCCL-EP runtime path implementation, no validation policy implementation, and
no fresh H200 fused-success evidence. PR #166 merged this scope as
`42b996666e279024b43f490a310c490a591a897d`.

## Validation Policy Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-validation-policy-map`.

PR #168 maps only the private validation policy required before a coordinator
may consume the PR #164 same-invocation request args and PR #166 UCCL-EP
capability metadata.

The validation policy remains private to the CUDA persistent-device runtime
path. It validates PR #164 same-invocation request args and PR #166 capability
metadata together before a coordinator can consume either dependency.

Failure ownership is explicit: missing metadata is unsupported, stale metadata
is failed, mismatched-rank metadata is failed, and mismatched-world-size
metadata is failed. descriptor-vocabulary mismatch is failed because
descriptor vocabulary must match dispatch/combine payload terms.
transport-mode mismatch is failed because transport mode must be `ep`.
adapter-provenance mismatch is failed because adapter provenance handles must
match the private capability id, invocation id, and rank/device map.
public/API-sourced metadata is failed as fabricated or untrusted pass
evidence.

This slice has no descriptor allocation policy implementation, no UCCL-EP
runtime dispatch, no coordinator implementation, no pass evidence, and no
H200 fused-success evidence. It must not allocate descriptors, implement
UCCL-EP runtime dispatch, construct the coordinator, change CUDA runtime
behavior, or claim pass evidence. Public `TaskArgs`, public `CallConfig`,
common runtime C API fields, UCCL host-runtime ABI fields, example JSON,
adapter provenance, and handoff metadata remain forbidden pass-evidence
paths.

PR #168 merged this validation policy scope as
`e33d232deccdf947b9c382a3605191d0d5ae0004`.

## Descriptor Allocation Policy Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map`.

This dependency slice maps only the private descriptor allocation policy
required before a later coordinator may allocate host-control records or
device-visible dispatch/combine descriptors. It does not allocate those
records or descriptors.

The descriptor allocation policy remains private to the CUDA
persistent-device runtime path. It preserves PR #164 same-invocation request
args, PR #166 UCCL-EP capability metadata, and PR #168 validation policy as
prerequisites rather than pass evidence.

The allocator owner is the future private
`persistent_device_uccl_ep_runtime_fusion` coordinator inside one CUDA
persistent-device runtime run context. The host-control record policy defines
private per-invocation records for invocation id, persistent graph descriptor
id, UCCL capability id, validated rank/device map, descriptor vocabulary,
allocation state, runtime owner, and shared ownership token slot. These are
not fields in public `TaskArgs`, public `CallConfig`, the common runtime C
API, the UCCL host-runtime ABI, example JSON, adapter provenance, or handoff
metadata.

The device-visible descriptor buffer policy defines future coordinator-owned
buffers allocated through the CUDA persistent-device runtime allocator and
visible only to the persistent-device scheduler and UCCL-EP runtime path.
The dispatch descriptor identity is the validated graph descriptor id,
capability id, invocation id, rank/device map, dispatch vocabulary, payload
shape, and coordinator-issued shared token. The combine descriptor identity
uses the same validated ids, rank/device map, combine vocabulary, payload
shape, and the same shared token as dispatch.

Failure ownership is explicit: missing policy is unsupported, stale policy is
failed, non-runtime-owned allocation is failed, descriptor-vocabulary mismatch
is failed, token-sharing mismatch is failed, rank/device mismatch is failed,
and public/API-sourced policy fields are failed as fabricated or untrusted
pass evidence. The shared-token requirement is strict: dispatch and combine
descriptors must carry one coordinator-issued token. The allocation lifetime
failure ownership belongs to that same private runtime owner.

This slice must not implement descriptor allocation, UCCL-EP runtime
dispatch, construct a coordinator, change CUDA runtime behavior, claim pass
evidence, or claim H200 fused-success evidence. Public `TaskArgs`, public
`CallConfig`, common runtime C API fields, UCCL host-runtime ABI fields,
example JSON, adapter provenance, and handoff metadata remain forbidden
pass-evidence paths.

PR #170 merged this descriptor allocation policy scope as
`bd0b59ee8d5afc969020d3aea047aafc9f3152be`.

## UCCL-EP Runtime Path Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map`.

This accepted dependency slice maps only the private UCCL-EP runtime path
required after PR #170's descriptor allocation policy and before UCCL-EP
runtime dispatch implementation, coordinator implementation, pass evidence,
or H200 fused-success evidence. It does not implement UCCL-EP runtime
dispatch.

The UCCL-EP runtime path remains private to the CUDA persistent-device
runtime path. It preserves PR #164 same-invocation request args, PR #166
UCCL-EP capability metadata, PR #168 validation policy, and PR #170
descriptor allocation policy as prerequisites rather than pass evidence.

The runtime-path owner is the future private
`persistent_device_uccl_ep_runtime_fusion` coordinator inside one CUDA
persistent-device runtime run context. The dispatch descriptor handoff uses
the PR #170 dispatch descriptor identity, including the coordinator-issued
shared token. The combine descriptor handoff uses the PR #170 combine
descriptor identity and must carry the same shared token as dispatch.
The map defines descriptor-token checks, rank/device checks, and
transport-mode checks before either descriptor handoff may be consumed.

The dispatch handoff identity includes invocation id, persistent graph
descriptor id, UCCL capability id, validated rank/device map, descriptor
vocabulary, dispatch payload shape, and coordinator-issued shared token. The
combine handoff identity includes the matching invocation id, graph descriptor
id, UCCL capability id, rank/device map, descriptor vocabulary, combine
payload shape, and exactly the same token. Public/API-sourced runtime-path
fields cannot replace any of those private inputs.

descriptor-token checks fail unless dispatch and combine descriptor views
carry the same coordinator-issued token from the current same-invocation
request. Rank/device checks fail unless persistent graph descriptor metadata,
private UCCL-EP capability metadata, and Worker-local CUDA device ordering
agree. Transport-mode checks fail unless the private capability metadata
declares `transport mode: ep`.

Runtime-path failure ownership is explicit: missing runtime path is
unsupported, stale descriptor views are failed, descriptor-token mismatch is
failed, rank/device mismatch is failed, transport-mode mismatch is failed,
descriptor-vocabulary mismatch is failed, and public/API-sourced runtime-path
fields are failed as fabricated or untrusted pass evidence.

This accepted slice must not implement UCCL-EP runtime dispatch, construct a
coordinator, allocate descriptors, change CUDA runtime behavior, claim pass
evidence, or claim H200 fused-success evidence. Public `TaskArgs`, public
`CallConfig`, common runtime C API fields, UCCL host-runtime ABI fields,
example JSON, adapter provenance, and handoff metadata remain forbidden
pass-evidence paths.

PR #172 accepted only this runtime-path map scope as
`21b2b32a475dc04e19700115af74510daef70859`: the private
UCCL-EP runtime path dependency map: runtime-path owner, dispatch descriptor
handoff, combine descriptor handoff, descriptor-token checks, rank/device
checks, transport-mode checks, and runtime-path failure ownership.

## Accepted UCCL-EP Runtime Path Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl`.

This implementation slice is limited to the private UCCL-EP runtime path
scaffold required after PR #172. It makes the mapped runtime path visible
to private CUDA persistent-device runtime code, but it must keep missing
coordinator behavior and missing descriptor allocation as unsupported or
failed states.

The slice must consume the PR #164 same-invocation request args, PR #166
capability metadata, PR #168 validation policy, PR #170 descriptor allocation
policy, and PR #172 runtime-path map as prerequisites rather than pass
evidence. Public `TaskArgs`, public `CallConfig`, common runtime C API
fields, UCCL host-runtime ABI fields, example JSON, adapter provenance, and
handoff metadata remain forbidden pass-evidence paths.

The slice must not implement the runtime-fusion coordinator, implement
descriptor allocation, claim pass evidence, claim H200 fused-success evidence,
or report `persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

The private runtime-path scaffold is limited to versioned descriptor views,
same-invocation id checks, coordinator-owned source checks, rank/device
checks, transport-mode checks, descriptor-vocabulary checks, and explicit
failed states for stale descriptor views or fabricated runtime-path fields.

PR #174 accepted this runtime-path scaffold scope as
`3b4b19a04855d27289fb9cdad802fee0c47d8265`. It accepted
`PtoCudaUcclEpRuntimePath`, `PtoCudaUcclEpRuntimeDescriptorView`, private
descriptor-view validation, and invocation-id propagation through private
CUDA runtime-fusion request state only. It did not implement the
runtime-fusion coordinator, descriptor allocation, UCCL-EP runtime dispatch,
pass evidence, fresh H200 fused-success evidence, public `TaskArgs`, public
`CallConfig`, common runtime C API fields, UCCL host-runtime ABI fields,
serving, vLLM, DeepSeek, throughput, or latency evidence.

## Accepted Descriptor Allocation Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl`.

This implementation slice is limited to private descriptor allocation
mechanics after the PR #174 runtime-path scaffold. It implements only the
private host-control record and device-visible dispatch/combine descriptor
buffer required by the PR #170 allocation policy, bound to the same
invocation id that PR #174 carries through runtime-fusion request state.

Implementation evidence is private to the CUDA runtime-fusion ABI and host
runtime path:

- `PtoCudaUcclEpDescriptorHostControl` and
  `PtoCudaUcclEpDeviceDescriptorBuffer` define the host-control record plus
  device-visible dispatch/combine descriptor buffer.
- `PtoCudaUcclEpDescriptorAllocation` binds those records to the PR #174
  `PtoCudaUcclEpRuntimePath` descriptor views.
- `pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors` creates the
  allocation from same-invocation private request state.
- `CudaDeviceRunner::record_runtime_fusion_unsupported` wires the allocation
  into the private request while keeping the final entry result unsupported
  because coordinator construction and UCCL-EP runtime dispatch are absent.

The slice is explicitly narrower than runtime-fusion coordinator construction
and narrower than UCCL-EP runtime dispatch. Missing coordinator behavior and
missing UCCL-EP runtime dispatch must remain unsupported or failed states.
Public `TaskArgs`, public `CallConfig`, common runtime C API fields, UCCL
host-runtime ABI fields, example JSON, adapter provenance, handoff metadata,
and payload provenance remain forbidden pass-evidence paths.

The slice must not claim pass evidence, fresh H200 fused-success evidence,
`persistent_device_uccl_ep_runtime_fusion.status: passed`,
`actual_fused_cross_gpu_execution: true`, RDMA, multi-node, serving, vLLM,
DeepSeek, throughput, or latency.

PR #176 accepted this descriptor allocation scaffold only, merged as
`6e0cecc174ae9db47573c4c0f1698be7accb295c`. It accepted the private
host-control record, device-visible dispatch/combine descriptor buffer,
allocation bundle, same-invocation binding, and private runtime-path handoff
into request state. It did not implement coordinator construction, UCCL-EP
runtime dispatch, pass evidence, fresh H200 fused-success evidence, public
`TaskArgs`, public `CallConfig`, common runtime C API fields, UCCL
host-runtime ABI fields, examples, stable docs, serving, vLLM, DeepSeek,
throughput, or latency.

## Accepted Runtime Fusion Coordinator Scaffold Status Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status`.

PR #178 merged as `aea89cc9dea8560602c72f84e5ff6e78ca526434` and accepted
only this private coordinator-construction scaffold/status slice. It defines
and wires private coordinator state needed to own the PR #176 descriptor
allocation and PR #174 runtime path for one private `ChipWorker::run`
invocation, but it stays narrower than UCCL-EP runtime dispatch and narrower
than pass evidence.

The coordinator scaffold owns the same invocation id, the accepted descriptor
allocation, the private runtime path, unsupported/failure status, and the
runtime-owned output sink. The focused private-entry test proves the
`missing_coordinator` failure clears only when that coordinator-shaped state is
present; UCCL-EP runtime dispatch remains absent and the final status remains
`unsupported`.

It cannot claim fused success until real UCCL-EP runtime dispatch and fresh
H200 fused-boundary evidence exist.

## Accepted Runtime Dispatch Scaffold Status Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status`.

This slice adds only a private UCCL-EP runtime-dispatch
scaffold/status gate from coordinator-owned state. It may consume the PR #178
coordinator-owned descriptor allocation and runtime path, check whether the
private invocation is dispatch-eligible, and write unsupported or failed
status through the runtime-owned output sink.

The implemented gate is private coordinator-owned state, not UCCL-EP runtime
dispatch. It records dispatch-scaffold eligibility for the same invocation.
If the gate is absent, the private result records
`missing_runtime_dispatch_scaffold` and `failed`; if the gate is present and
eligible, the result remains `unsupported`.

It must not run real UCCL-EP dispatch/combine work, claim UCCL-EP runtime
dispatch success, emit scheduler/runtime pass evidence, claim fresh H200
fused success, expand public APIs or UCCL host-runtime ABI fields, or report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

PR #180 merged as `dc32c52dfccfd7838f865a11c3d4837e8ee568ba` and accepted
only this private coordinator-owned gate. Missing gate state yields
`missing_runtime_dispatch_scaffold` and a failed private result; an eligible
prepared gate remains `unsupported`; output is mirrored to the runtime-owned
sink. It does not provide real UCCL-EP dispatch/combine work,
scheduler/runtime pass evidence, or H200 fused success.

## Runtime Dispatch Request Handoff Map Slice

Selected branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map`.

This runtime dispatch request/driver handoff map is a docs/test dependency
map. It starts after PR #181 at
`05457b7dead2f561be22c24c72771add880f4562`. It defines only the private
UCCL-EP runtime dispatch request/driver handoff map needed after the PR #180
scaffold/status gate and before any real runtime dispatch.

Request owner is the private `PtoCudaRuntimeFusionCoordinator`, which may
form a future handoff request only from coordinator-owned descriptor
allocation, private runtime path, validation policy, capability metadata,
same invocation id, PR #180 gate status, and runtime-owned output sink.
Driver owner is a future private UCCL-EP runtime dispatch driver below the
CUDA persistent-device runtime path; it cannot source driver state from
public TaskArgs, public CallConfig, common runtime C API, or UCCL
host-runtime ABI fields.

Status dependency is the PR #180 runtime-dispatch scaffold/status gate:
missing gate yields `missing_runtime_dispatch_scaffold` and a failed private
result; an eligible prepared gate remains `unsupported`; output is mirrored
to the runtime-owned sink. Failure ownership remains coordinator-owned until
a later private driver scaffold accepts the handoff.

Unsupported handoff state covers missing prepared gate, missing request
fields, or missing private driver. Failed handoff state covers stale
invocation id, rank/device mismatch, descriptor-token mismatch, failed
scaffold/status gate, public/API-sourced handoff fields, or fabricated pass
evidence.

The slice records no UCCL-EP dispatch/combine work, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public TaskArgs, no public
CallConfig, no common runtime C API, no UCCL host-runtime ABI, no examples,
no stable docs, and no performance claims. It must not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status`.
It is exactly one next PR-sized implementation slice and may add only
private request/driver handoff scaffold/status plumbing. It is narrower than
pass evidence and must not run real UCCL-EP dispatch/combine work.

## Runtime Dispatch Request Handoff Scaffold Status Slice

Selected branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status`.

PR #183 merged as `80b6606282956f38ca6c9a3c52c95d0e5e3a457f` and accepted
only private request/driver handoff scaffold/status plumbing for the PR #182
map. The handoff remains below the CUDA persistent-device runtime path and
below the PR #180 coordinator-owned runtime-dispatch scaffold/status gate.
`PtoCudaRuntimeFusionCoordinator` owns the request and the private
driver-state placeholder; no public
`TaskArgs`, public `CallConfig`, common runtime C API, or UCCL host-runtime
ABI field owns or supplies this state.

The private scaffold validates same invocation id, coordinator-owned runtime
path, prepared PR #180 gate state, request owner, private driver-state
pointer, and runtime-owned output sink. Missing or stale private driver state
records `missing_runtime_dispatch_handoff_driver` and `failed`; a valid
handoff scaffold records `unsupported` with `unsupported_boundary`. It never
sets `actual_fused_cross_gpu_execution` true and never reports
`persistent_device_uccl_ep_runtime_fusion.status: passed`.

This is not UCCL-EP dispatch/combine work, scheduler/runtime pass evidence,
fresh H200 fused success, public API expansion, an example, a stable-doc
claim, or a performance claim.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map`. It is
exactly one next PR-sized docs/test dependency slice for private driver-owned
unsupported/failed status vocabulary and failure ownership after this
handoff scaffold, still without real dispatch/combine work or pass evidence.

## Post-PR183 Status Refresh

Selected branch:
`nvidia-goal-status-post-runtime-dispatch-handoff-scaffold`.

This status refresh records PR #183 as accepted only for the private
request/driver handoff scaffold/status path: private ABI state under
`PtoCudaRuntimeFusionCoordinator`, same invocation id, coordinator-owned
runtime path/gate, request owner, private driver-state pointer,
runtime-owned output sink, missing/stale handoff driver failure, and valid
handoff remaining `unsupported`. It keeps
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map` as the
single next PR-sized dependency slice and does not claim real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, or H200 fused
success.

## Runtime Dispatch Driver Status Map Slice

Selected branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map`.

This docs/test dependency slice maps only private driver-owned
unsupported/failed status vocabulary and failure ownership after PR #183.
PR #183 merged as `80b6606282956f38ca6c9a3c52c95d0e5e3a457f` and accepted
only the request/driver handoff scaffold/status path.

The private driver owner is the future UCCL-EP runtime dispatch driver below
the CUDA persistent-device runtime path. The failure owner boundary keeps
missing driver remains handoff-owned failed until a driver accepts the
handoff, while stale accepted driver is driver-owned failed after driver
acceptance. A valid handoff remains `unsupported` because there is still no
real dispatch or combine backend.

Unsupported driver statuses are `driver_missing`, `driver_stale`,
`driver_not_bound_to_handoff`, `driver_no_dispatch_backend`,
`driver_no_combine_backend`, and `driver_unsupported_boundary`.
Failed driver statuses are `driver_owner_mismatch`,
`driver_invocation_mismatch`, `driver_runtime_path_mismatch`,
`driver_descriptor_token_mismatch`, `driver_rank_device_mismatch`,
`driver_status_sink_mismatch`, `driver_public_api_sourced_state`, and
`driver_fabricated_pass_evidence`.

This selection still records no real UCCL-EP dispatch/combine work, no
scheduler/runtime pass evidence, no fresh H200 fused success, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples, stable docs, or performance claims. It
does not report `persistent_device_uccl_ep_runtime_fusion.status: passed` or
set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-scaffold-status`.
It may add only a private driver scaffold/status owner for this vocabulary,
without real UCCL-EP dispatch/combine work or pass evidence.

## Runtime Dispatch Driver Scaffold Status Slice

Selected branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-scaffold-status`.

This implementation slice accepts only private driver scaffold/status
ownership for the PR #185 status vocabulary after PR #185 merged as
`8619767d0eacb5c870b6a56337c6bcb380a2af75`. The code surface stays in the
private CUDA runtime-fusion ABI:
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_status_name`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`.

A valid prepared driver scaffold is bound to the PR #183 handoff status,
private driver state, coordinator-owned runtime path, same invocation id, and
runtime-owned output sink. It remains `unsupported` and records
`driver_unsupported_boundary`; `actual_fused_cross_gpu_execution` remains
false and `persistent_device_uccl_ep_runtime_fusion.status` is not `passed`.

Malformed, stale, or mismatched private driver scaffold/status records a
failed private result with driver-owned failures such as
`driver_owner_mismatch`, `driver_invocation_mismatch`,
`driver_runtime_path_mismatch`, `driver_descriptor_token_mismatch`,
`driver_rank_device_mismatch`, `driver_status_sink_mismatch`,
`driver_public_api_sourced_state`, and
`driver_fabricated_pass_evidence`.

Selected next slice:
`nvidia-goal-status-post-runtime-dispatch-driver-scaffold-status`. It is a
review-facing status refresh only; real UCCL-EP dispatch/combine work and pass
evidence remain out of scope.

## Post-Runtime-Dispatch-Driver-Scaffold Status Refresh

Branch:
`nvidia-goal-status-post-runtime-dispatch-driver-scaffold-status`.
PR #186 merged as `7589e2df44ad4df9c200cd4ec673dacac0a27a71`
(`Add runtime dispatch driver scaffold status`) and is accepted only for
private runtime-dispatch driver scaffold/status ownership.

The accepted private surface is
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`, driver-owned failure bits,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`,
and the host runtime private call. The valid status remains unsupported with
`driver_unsupported_boundary`; malformed/mismatched produces failed private
result with driver-owned failure names.

This status refresh records no real UCCL-EP dispatch/combine work, no
scheduler/runtime pass evidence, no fresh H200 fused success, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples, stable docs, or performance claims. It
does not report `persistent_device_uccl_ep_runtime_fusion.status: passed` and
does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-map`.
This is selected exactly one next PR-sized dependency map slice for real
runtime dispatch driver request/backend ownership. It is not
implementation/pass evidence and does not authorize real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, or fresh H200 fused
success.

## Runtime Dispatch Driver Backend Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-map`.

This dependency map records the future private runtime dispatch driver's
request/backend ownership boundary after PR #186
(`7589e2df44ad4df9c200cd4ec673dacac0a27a71`). It follows the accepted
private driver scaffold/status surface:
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`, driver-owned failure bits, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`.
The valid prepared driver scaffold remains `unsupported`.

Boundary ownership:

- private driver request owner: the future private driver owns the request
  only after the PR #186 scaffold matches the handoff, invocation id, runtime
  path, descriptor token, rank/device map, and runtime-owned output sink;
- dispatch backend placeholder: the future UCCL-EP dispatch backend remains a
  private driver placeholder with no transport, payload transfer, scheduler
  transition, or pass evidence;
- combine backend placeholder: the future UCCL-EP combine backend remains a
  private driver placeholder with no reduce/combine transport, payload
  release, scheduler transition, or pass evidence;
- status sink owner: the runtime-owned output sink remains the review-facing
  sink; example JSON, adapter-only provenance, public `TaskArgs`, public
  `CallConfig`, common runtime C API, and UCCL host-runtime ABI fields do not
  own or source backend status;
- driver-owned failure propagation: after driver acceptance, backend
  request/backend/status-sink mismatches propagate as driver-owned failed
  states.

Unsupported states are `driver_backend_request_unbound`,
`driver_dispatch_backend_placeholder`, `driver_combine_backend_placeholder`,
`driver_status_sink_unbound`, and
`driver_backend_map_unsupported_boundary`.

Failed states are `driver_backend_owner_mismatch`,
`driver_backend_invocation_mismatch`,
`driver_backend_runtime_path_mismatch`,
`driver_backend_descriptor_token_mismatch`,
`driver_backend_rank_device_mismatch`,
`driver_backend_status_sink_mismatch`,
`driver_backend_public_api_sourced_state`, and
`driver_backend_fabricated_pass_evidence`.

The invalid pass-evidence boundary rejects public API state, adapter metadata,
example JSON, and hand-authored review artifacts as evidence for the driver
request/backend boundary. This slice records no real UCCL-EP dispatch/combine
work, no scheduler/runtime pass evidence, no fresh H200 fused success, no
examples, stable docs, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
driver backend scaffold/status only.

## Runtime Dispatch Driver Backend Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-scaffold-status`.

This private implementation slice follows PR #188
(`7bc598f75d5738193a7b53fa10a751f2518edb17`). It keeps backend status as
runtime-private scaffold evidence for future UCCL-EP driver work, not as
transport selection, scheduler/runtime pass evidence, or H200 fused success.
It does not add public `TaskArgs`, public `CallConfig`, common runtime C API,
UCCL host-runtime ABI, examples, stable docs, or performance claims.

Implementation evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_scaffold_status_is_driver_owned`.

The valid prepared backend scaffold/status remains `unsupported`,
`actual_fused_cross_gpu_execution` remains `0`, no passed status is reported,
and no dispatch backend or combine backend execution is performed.

Unsupported backend-scaffold states are
`driver_backend_request_unbound`, `driver_dispatch_backend_placeholder`,
`driver_combine_backend_placeholder`, `driver_status_sink_unbound`, and
`driver_backend_map_unsupported_boundary`.

Failed backend-scaffold states are `driver_backend_owner_mismatch`,
`driver_backend_invocation_mismatch`,
`driver_backend_runtime_path_mismatch`,
`driver_backend_descriptor_token_mismatch`,
`driver_backend_rank_device_mismatch`,
`driver_backend_status_sink_mismatch`,
`driver_backend_public_api_sourced_state`, and
`driver_backend_fabricated_pass_evidence`.

The focused red check failed first because
`runtime_dispatch_driver_backend_scaffold_status`,
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`
were missing. The slice selects
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-map`
as exactly one next PR-sized dependency map slice. It records no real
UCCL-EP dispatch/combine work, no scheduler/runtime pass evidence, no fresh
H200 fused success, no `persistent_device_uccl_ep_runtime_fusion.status:
passed`, and no `actual_fused_cross_gpu_execution: true`.

## Runtime Dispatch Driver Backend Request Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-map`.

This docs/test dependency map records the future private driver backend
request after PR #189 (`707cc81a818fdc00e4f592acb2f17538d1f6eb0a`). It
consumes the accepted private backend scaffold/status input:
`PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendStatus`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`.
The valid backend scaffold/status remains `unsupported`; it is not a backend
request, dispatch request, combine request, or pass-evidence source.

Boundary ownership:

- private backend request owner: the future private request owner accepts the
  PR #189 backend scaffold/status input only after backend owner, invocation
  id, runtime path, descriptor token, rank/device map, world size, and
  runtime-owned status sink match;
- dispatch request placeholder: private driver-owned request shape for future
  UCCL-EP dispatch, with no transport, payload transfer, kernel launch,
  scheduler transition, or pass evidence;
- combine request placeholder: private driver-owned request shape for future
  UCCL-EP combine, with no reduce/combine transport, payload release, kernel
  launch, scheduler transition, or pass evidence;
- descriptor token validation: the request must reuse the accepted backend
  scaffold/status descriptor token;
- rank/device validation: the request rank/device map must match the accepted
  scaffold rank, CUDA device, and world-size metadata;
- status sink ownership: the runtime-owned output sink remains the only
  review-facing status sink;
- invalid public/provenance sources: example JSON, adapter-only provenance,
  public `TaskArgs`, public `CallConfig`, common runtime C API fields,
  UCCL host-runtime ABI fields, and hand-authored review artifacts cannot
  source backend request state.

Unsupported backend-request states are `driver_backend_request_pending`,
`driver_backend_dispatch_request_placeholder`,
`driver_backend_combine_request_placeholder`,
`driver_backend_request_status_sink_unbound`, and
`driver_backend_request_map_unsupported_boundary`.

Failed backend-request states are `driver_backend_request_owner_mismatch`,
`driver_backend_request_invocation_mismatch`,
`driver_backend_request_runtime_path_mismatch`,
`driver_backend_request_descriptor_token_mismatch`,
`driver_backend_request_rank_device_mismatch`,
`driver_backend_request_status_sink_mismatch`,
`driver_backend_request_public_api_sourced_state`,
`driver_backend_request_provenance_sourced_state`, and
`driver_backend_request_fabricated_pass_evidence`.

The invalid public/provenance sources boundary keeps public API state,
adapter-only provenance, example JSON, common runtime C API fields, UCCL
host-runtime ABI fields, and hand-authored review artifacts out of the
backend request. This slice records no real UCCL-EP dispatch/combine work,
no scheduler/runtime pass evidence, no fresh H200 fused success, no examples,
stable docs, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
backend request scaffold/status only.

## Runtime Dispatch Driver Backend Request Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-request-scaffold-status`.

This private implementation slice follows PR #190
(`4223edd9fa3c5e58b62eff1d7c27b1a54670766d`). It keeps backend request
status as runtime-private scaffold evidence for future UCCL-EP driver work,
not as transport selection, scheduler/runtime pass evidence, or H200 fused
success. It does not add public `TaskArgs`, public `CallConfig`, common
runtime C API, UCCL host-runtime ABI, examples, stable docs, or performance
claims.

Implementation evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_request_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_request_scaffold_status_is_backend_owned`.

The valid prepared backend request scaffold/status remains `unsupported`,
`actual_fused_cross_gpu_execution` remains `0`, no passed status is reported,
and no dispatch request or combine request execution is performed.

Unsupported backend-request scaffold states are
`driver_backend_request_pending`,
`driver_backend_dispatch_request_placeholder`,
`driver_backend_combine_request_placeholder`,
`driver_backend_request_status_sink_unbound`, and
`driver_backend_request_map_unsupported_boundary`.

Failed backend-request scaffold states are
`driver_backend_request_owner_mismatch`,
`driver_backend_request_invocation_mismatch`,
`driver_backend_request_runtime_path_mismatch`,
`driver_backend_request_descriptor_token_mismatch`,
`driver_backend_request_rank_device_mismatch`,
`driver_backend_request_status_sink_mismatch`,
`driver_backend_request_public_api_sourced_state`,
`driver_backend_request_provenance_sourced_state`, and
`driver_backend_request_fabricated_pass_evidence`.

The focused red check failed first because the private backend request
scaffold/status owner symbols were missing. The focused green check passed
with `1 passed in 0.42s`; full private-entry pytest passed with
`19 passed in 5.11s`. This slice records no real UCCL-EP dispatch/combine
work, no scheduler/runtime pass evidence, no fresh H200 fused success, no
`persistent_device_uccl_ep_runtime_fusion.status: passed`, and no
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-map`.
This is selected exactly one next PR-sized dependency map slice for the
future private dispatch request placeholder.

## Runtime Dispatch Driver Backend Dispatch Request Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-map`.

This docs/test dependency map records the future private dispatch request
placeholder after PR #191 (`d4cbbfc130b356d90b649aa40f2c904d0fc8a081`). It
consumes the accepted private backend request scaffold/status input:
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

Unsupported dispatch-request states are
`driver_backend_dispatch_request_pending`,
`driver_backend_dispatch_payload_descriptor_placeholder`,
`driver_backend_dispatch_output_status_sink_unbound`,
`driver_backend_dispatch_request_map_unsupported_boundary`, and
`driver_backend_dispatch_payload_transfer_unimplemented`.

Failed dispatch-request states are
`driver_backend_dispatch_request_owner_mismatch`,
`driver_backend_dispatch_request_invocation_mismatch`,
`driver_backend_dispatch_request_scaffold_mismatch`,
`driver_backend_dispatch_request_descriptor_token_mismatch`,
`driver_backend_dispatch_request_rank_device_mismatch`,
`driver_backend_dispatch_request_status_sink_mismatch`,
`driver_backend_dispatch_request_public_api_sourced_state`,
`driver_backend_dispatch_request_provenance_sourced_state`, and
`driver_backend_dispatch_request_fabricated_pass_evidence`.

The invalid public/provenance sources boundary keeps public API state,
adapter-only provenance, example JSON, common runtime C API fields, UCCL
host-runtime ABI fields, and hand-authored review artifacts out of the
dispatch request. This slice records no real UCCL-EP dispatch/combine work,
no scheduler/runtime pass evidence, no fresh H200 fused success, no examples,
stable docs, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
dispatch request scaffold/status only.

## Runtime Dispatch Driver Backend Dispatch Request Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-dispatch-request-scaffold-status`.

This private implementation slice follows PR #192
(`14aaedd8865ea7351cd30ee1a0dc46804b7d0f36`). It adds private
dispatch-request scaffold/status vocabulary for the UCCL-EP runtime-fusion
boundary:
`PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_dispatch_request_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_dispatch_request_scaffold_status`,
and `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD`.

The scaffold consumes the existing
`PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus` input and
keeps dispatch request status private to the coordinator-owned runtime-fusion
state. A valid prepared dispatch request scaffold/status remains
`unsupported`; `actual_fused_cross_gpu_execution` remains `0`; malformed
owner, invocation, request scaffold, descriptor token, rank/device, or status
sink state fails through the private result sink.

Unsupported states are `driver_backend_dispatch_request_pending`,
`driver_backend_dispatch_payload_descriptor_placeholder`,
`driver_backend_dispatch_output_status_sink_unbound`,
`driver_backend_dispatch_request_map_unsupported_boundary`, and
`driver_backend_dispatch_payload_transfer_unimplemented`.

Failed states are `driver_backend_dispatch_request_owner_mismatch`,
`driver_backend_dispatch_request_invocation_mismatch`,
`driver_backend_dispatch_request_scaffold_mismatch`,
`driver_backend_dispatch_request_descriptor_token_mismatch`,
`driver_backend_dispatch_request_rank_device_mismatch`,
`driver_backend_dispatch_request_status_sink_mismatch`,
`driver_backend_dispatch_request_public_api_sourced_state`,
`driver_backend_dispatch_request_provenance_sourced_state`, and
`driver_backend_dispatch_request_fabricated_pass_evidence`.

The focused private-entry evidence is
`test_private_runtime_dispatch_driver_backend_dispatch_request_scaffold_status_is_backend_owned`.
This slice adds no real UCCL-EP dispatch/combine work, no scheduler/runtime
pass evidence, no fresh H200 fused success, no public `TaskArgs`, no public
`CallConfig`, no common runtime C API, no UCCL host-runtime ABI, no examples,
stable docs, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.
Focused red check failed first with `1 failed in 0.41s`; focused green check
passed with `1 passed in 0.41s`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine request placeholder.

## Runtime Dispatch Driver Backend Combine Request Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-map`.

This docs/test dependency map records the future private combine request
placeholder after PR #193 (`f969ea00c6858a6633ee53fd33bf77dd434097dc`).
PR #193 is accepted only for private backend dispatch request scaffold/status
vocabulary and evidence:
`PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_dispatch_request_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_dispatch_request_scaffold_status`,
and `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD`.

The future combine request placeholder consumes the backend request
scaffold/status input and the dispatch request scaffold/status dependency.
Both remain private unsupported prerequisites and cannot source combine
request state, scheduler/runtime pass evidence, or H200 fused success.

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

Unsupported combine-request states are `driver_backend_combine_request_pending`,
`driver_backend_combine_payload_descriptor_placeholder`,
`driver_backend_combine_output_status_sink_unbound`,
`driver_backend_combine_request_map_unsupported_boundary`, and
`driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-request states are
`driver_backend_combine_request_owner_mismatch`,
`driver_backend_combine_request_invocation_mismatch`,
`driver_backend_combine_request_scaffold_mismatch`,
`driver_backend_combine_request_descriptor_token_mismatch`,
`driver_backend_combine_request_rank_device_mismatch`,
`driver_backend_combine_request_status_sink_mismatch`,
`driver_backend_combine_request_public_api_sourced_state`,
`driver_backend_combine_request_provenance_sourced_state`, and
`driver_backend_combine_request_fabricated_pass_evidence`.

The map unsupported boundary and payload transfer unimplemented vocabulary
are future placeholders only. This slice records no real UCCL-EP
dispatch/combine work, no scheduler/runtime pass evidence, no fresh H200
fused success, no public `TaskArgs`, no public `CallConfig`, no common
runtime C API, no UCCL host-runtime ABI, and no examples, stable docs,
serving, vLLM, DeepSeek, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
combine request scaffold/status only.

## Runtime Dispatch Driver Backend Combine Request Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-request-scaffold-status`.

This slice implements private combine request scaffold/status after PR #194
(`562778f051ca87cf3f62d796860a8fd4c3476a32`) while keeping communication
selection at the unsupported private ABI boundary.

Implementation evidence in
`src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h`:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_request_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_request_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD`.

Focused private-entry coverage is
`test_private_runtime_dispatch_driver_backend_combine_request_scaffold_status_is_backend_owned`.
The valid prepared combine request scaffold/status remains `unsupported`;
`actual_fused_cross_gpu_execution` remains `0`; no passed status is reported.
focused red check failed first with `1 failed in 0.42s`; focused green check
passed with `1 passed in 0.42s`.

Unsupported combine-request states are `driver_backend_combine_request_pending`,
`driver_backend_combine_payload_descriptor_placeholder`,
`driver_backend_combine_output_status_sink_unbound`,
`driver_backend_combine_request_map_unsupported_boundary`, and
`driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-request states are
`driver_backend_combine_request_owner_mismatch`,
`driver_backend_combine_request_invocation_mismatch`,
`driver_backend_combine_request_scaffold_mismatch`,
`driver_backend_combine_request_descriptor_token_mismatch`,
`driver_backend_combine_request_rank_device_mismatch`,
`driver_backend_combine_request_status_sink_mismatch`,
`driver_backend_combine_request_public_api_sourced_state`,
`driver_backend_combine_request_provenance_sourced_state`, and
`driver_backend_combine_request_fabricated_pass_evidence`.

This slice records no real UCCL-EP dispatch/combine work, no scheduler/runtime
pass evidence, no fresh H200 fused success, no public `TaskArgs`, no public
`CallConfig`, no common runtime C API, no UCCL host-runtime ABI, and no
examples, stable docs, serving, vLLM, DeepSeek, or performance claims. It
does not report `persistent_device_uccl_ep_runtime_fusion.status: passed` and
does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine payload descriptor placeholder.

## Runtime Dispatch Driver Backend Combine Payload Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-map`.

This docs/test dependency map records the future private combine payload
descriptor placeholder after PR #195
(`e09b67a7a00f481f8c9dd4508d1adc9e88030d00`). PR #195 is accepted only for
private backend combine request scaffold/status vocabulary and evidence:
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombineRequestStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_request_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_request_scaffold_status`,
and `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_REQUEST_SCAFFOLD`.

The future combine payload descriptor placeholder consumes the backend
request scaffold/status input, the dispatch request scaffold/status
dependency, and the combine request scaffold/status dependency. All three
remain private unsupported prerequisites and cannot source combine payload
descriptor state, scheduler/runtime pass evidence, or H200 fused success.

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

Unsupported combine-payload states are `driver_backend_combine_payload_pending`,
`driver_backend_combine_payload_descriptor_placeholder`,
`driver_backend_combine_payload_output_status_sink_unbound`,
`driver_backend_combine_payload_map_unsupported_boundary`, and
`driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-payload states are
`driver_backend_combine_payload_owner_mismatch`,
`driver_backend_combine_payload_invocation_mismatch`,
`driver_backend_combine_payload_request_scaffold_mismatch`,
`driver_backend_combine_payload_descriptor_token_mismatch`,
`driver_backend_combine_payload_rank_device_mismatch`,
`driver_backend_combine_payload_status_sink_mismatch`,
`driver_backend_combine_payload_public_api_sourced_state`,
`driver_backend_combine_payload_provenance_sourced_state`, and
`driver_backend_combine_payload_fabricated_pass_evidence`.

The map unsupported boundary and payload transfer unimplemented vocabulary
are future placeholders only. This slice records no real UCCL-EP
dispatch/combine work, no scheduler/runtime pass evidence, no fresh H200
fused success, no public `TaskArgs`, no public `CallConfig`, no common
runtime C API, no UCCL host-runtime ABI, and no examples, stable docs,
serving, vLLM, DeepSeek, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
combine payload descriptor scaffold/status only.

## Runtime Dispatch Driver Backend Combine Payload Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-scaffold-status`.

This selection note records private combine payload descriptor scaffold/status
after PR #196 (`af62d14d456b34fb1ef7fb2f9b4b6af7bc0bd4d1`). The slice adds
only private coordinator-owned payload descriptor status under the runtime
fusion ABI. It depends on the combine request scaffold/status and preserves
the dispatch/combine descriptor token, rank, CUDA device, world size, and
runtime-owned output/status sink validation shape.

Accepted private ABI/status vocabulary:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_combine_payload_scaffold_status_is_backend_owned`.

The valid prepared combine payload descriptor scaffold/status remains
`unsupported`; `actual_fused_cross_gpu_execution` remains `0`. Malformed
owner, invocation, combine-request dependency, descriptor token, rank/device,
status sink, public source, provenance source, or fabricated pass-evidence
state fails through the private entry instead of sourcing pass evidence.

Unsupported combine-payload states are `driver_backend_combine_payload_pending`,
`driver_backend_combine_payload_descriptor_placeholder`,
`driver_backend_combine_payload_output_status_sink_unbound`,
`driver_backend_combine_payload_map_unsupported_boundary`, and
`driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-payload states are
`driver_backend_combine_payload_owner_mismatch`,
`driver_backend_combine_payload_invocation_mismatch`,
`driver_backend_combine_payload_request_scaffold_mismatch`,
`driver_backend_combine_payload_descriptor_token_mismatch`,
`driver_backend_combine_payload_rank_device_mismatch`,
`driver_backend_combine_payload_status_sink_mismatch`,
`driver_backend_combine_payload_public_api_sourced_state`,
`driver_backend_combine_payload_provenance_sourced_state`, and
`driver_backend_combine_payload_fabricated_pass_evidence`.

Focused TDD evidence: focused red check failed first with
`1 failed in 0.44s`; focused green check passed with `1 passed in 0.43s`.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior, no payload transfer, no scheduler/runtime pass evidence,
no fresh H200 fused success, no public `TaskArgs`, no public `CallConfig`,
no common runtime C API, no UCCL host-runtime ABI, and no examples, stable
docs, serving, vLLM, DeepSeek, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine payload transfer boundary.

## Runtime Dispatch Driver Backend Combine Payload Transfer Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-map`.

This selection note maps the future private combine payload transfer boundary
after PR #197 (`3337516c95fcd5f6129c515585d92e3f95f0c444`). This is a
dependency map only and must not implement source behavior. PR #197 is
accepted only for private backend combine payload descriptor scaffold/status
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

Unsupported combine-payload-transfer states are
`driver_backend_combine_payload_transfer_pending`,
`driver_backend_combine_payload_transfer_descriptor_placeholder`,
`driver_backend_combine_payload_transfer_output_status_sink_unbound`,
`driver_backend_combine_payload_transfer_map_unsupported_boundary`, and
`driver_backend_combine_payload_transfer_unimplemented`.

Failed combine-payload-transfer states are
`driver_backend_combine_payload_transfer_owner_mismatch`,
`driver_backend_combine_payload_transfer_invocation_mismatch`,
`driver_backend_combine_payload_transfer_payload_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_provenance_sourced_state`, and
`driver_backend_combine_payload_transfer_fabricated_pass_evidence`.

The map unsupported boundary and payload transfer unimplemented vocabulary
are future placeholders only. This slice records no real UCCL-EP
dispatch/combine work, no descriptor allocation behavior, no payload transfer
implementation, no transport/backend execution, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public `TaskArgs`, no public
`CallConfig`, no common runtime C API, no UCCL host-runtime ABI, and no
examples, stable docs, serving, vLLM, DeepSeek, or performance claims. It
does not report `persistent_device_uccl_ep_runtime_fusion.status: passed` and
does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
combine payload transfer scaffold/status only.

## Runtime Dispatch Driver Backend Combine Payload Transfer Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-scaffold-status`.

This selection note records private combine payload transfer scaffold/status
after PR #198 (`227eae4c34a1182aab3548951380379da4582dc8`). It accepts only
private coordinator-owned transfer status in the CUDA runtime fusion ABI and
does not implement payload transfer source behavior.

Accepted private ABI/status vocabulary:

- `PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_SCAFFOLD_STATUS_VERSION`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status`;
- `test_private_runtime_dispatch_driver_backend_combine_payload_transfer_scaffold_status_is_backend_owned`.

The transfer scaffold/status depends on backend request scaffold/status input,
the dispatch request scaffold/status dependency, the combine request
scaffold/status dependency, and the combine payload descriptor
scaffold/status dependency. It validates owner, invocation, descriptor token,
rank/device, status sink, payload scaffold dependency, invalid
public/provenance sources, and fabricated pass evidence.

The current `PtoCudaRuntimeFusionFailure` mask is exhausted through
`1U << 31U`, so this slice reuses the existing combine-payload scaffold
aggregate failure bit,
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`,
instead of adding `1U << 32U` or widening public ABI fields.

The valid prepared combine payload transfer scaffold/status remains
`unsupported`; `actual_fused_cross_gpu_execution` remains `0`. The status
vocabulary includes `driver_backend_combine_payload_transfer_pending`,
`driver_backend_combine_payload_transfer_unimplemented`,
`driver_backend_combine_payload_transfer_output_status_sink_unbound`,
`driver_backend_combine_payload_transfer_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_owner_mismatch`,
`driver_backend_combine_payload_transfer_invocation_mismatch`,
`driver_backend_combine_payload_transfer_request_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_dispatch_request_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_combine_request_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_payload_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_rank_device_mismatch`,
`driver_backend_combine_payload_transfer_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_provenance_sourced_state`, and
`driver_backend_combine_payload_transfer_fabricated_pass_evidence`.

Focused TDD evidence: focused red check failed first with
`1 failed in 0.46s`; focused green check passed with `1 passed in 0.45s`.
Final verification passed: the final focused green rerun passed with `1 passed`; `git diff --check`
passed; targeted markdownlint over the five NVIDIA in-progress docs reported
`0 error(s)`; the NVIDIA review guard passed;
`test_cuda_runtime_fusion_private_entry.py` passed with `23 passed`; and
`test_nvidia_review_artifacts.py` passed with `85 passed`.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no
transport/backend execution, no scheduler/runtime pass evidence, no fresh
H200 fused success, no public API expansion, no public `TaskArgs`, no public
`CallConfig`, no common runtime C API, no UCCL host-runtime ABI, and no
examples, stable docs, serving, vLLM, DeepSeek, or performance claims. It
does not report `persistent_device_uccl_ep_runtime_fusion.status: passed` and
does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-map`.
This is selected exactly one next PR-sized dependency map slice for the future
private combine payload transfer completion boundary.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Map Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-map`
slice after PR #199
(`41c9c894ad511534d943180bccb10aab8fba3f7b`). This is a dependency map
only, not source behavior, for the future private combine payload transfer
completion boundary.

PR #199 is accepted only for private combine payload transfer scaffold/status
vocabulary and evidence:
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

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no transport/backend execution, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public API expansion, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples/stable docs/serving/vLLM/DeepSeek/performance claims.
It does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`
and does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
completion scaffold/status only.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Scaffold Status Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-scaffold-status`
slice after PR #200
(`47136ab0b1cff42b7ad3448809a3b9e5bf44db43`). This is a private
completion scaffold/status only, not payload transfer or completion
execution.

PR #200 is accepted as the private combine payload transfer completion
boundary map. This implementation adds private ABI/status vocabulary:
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status`,
and
`test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_scaffold_status_is_backend_owned`.

The completion scaffold/status consumes backend request scaffold/status
input, the dispatch request scaffold/status dependency, the combine request
scaffold/status dependency, the combine payload descriptor scaffold/status
dependency, and the combine payload transfer scaffold/status dependency.

Coordinator-owned validation fails closed for owner, invocation, transfer
scaffold dependency, descriptor token, rank/device, status sink, completion
sink, public/provenance sourced state, and fabricated pass evidence. The
valid prepared completion scaffold/status remains unsupported with
`actual_fused_cross_gpu_execution == 0`.

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

Failure-bit design note: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. Completion
scaffold/status validation reuses
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
as the existing combine-payload scaffold aggregate failure bit while keeping
completion status enum and status names unambiguous.

Red/green evidence: focused red check failed first with
`1 failed in 0.47s` because the completion scaffold/status API was missing;
focused green check passed with `1 passed in 0.44s` after adding the private
completion scaffold/status implementation.

Final verification passed: focused green passed;
`git diff --check` passed with no output; targeted `markdownlint-cli2`
reported `Summary: 0 error(s)`; the NVIDIA review guard reported
`nvidia review guard passed`; the full private runtime-fusion pytest reported
`24 passed`; and the NVIDIA review-artifact pytest reported `87 passed`.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no transport/backend execution, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public API expansion, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples/stable docs/serving/vLLM/DeepSeek/performance claims.
It does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`
and does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-map`.
This is selected exactly one next PR-sized docs/test map slice for the future
private combine payload transfer completion handoff boundary. It is not a
payload transfer implementation or completion implementation.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Map Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-map`
slice after PR #201 (`47e7bd1e`). This is a dependency map only, not source
behavior, for the future private combine payload transfer completion handoff
boundary.

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

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no handoff implementation, no transport/backend execution,
no scheduler/runtime pass evidence, no fresh H200 fused success, no public
API expansion, no public `TaskArgs`, no public `CallConfig`, no common
runtime C API, no UCCL host-runtime ABI, and no
examples/stable docs/serving/vLLM/DeepSeek/performance claims. It does not
report `persistent_device_uccl_ep_runtime_fusion.status: passed` and does
not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
handoff scaffold/status only.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Scaffold Status Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-scaffold-status`
slice after merged PR #202 (`925eed3e`). It implements private
scaffold/status vocabulary and coordinator-owned validation only.

PR #202 is accepted as the dependency map for this handoff boundary. The
implemented private vocabulary is
`PTO_CUDA_UCCL_EP_RUNTIME_DISPATCH_DRIVER_BACKEND_COMBINE_PAYLOAD_TRANSFER_COMPLETION_HANDOFF_SCAFFOLD_STATUS_VERSION`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendCombinePayloadTransferCompletionHandoffStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status`,
and
`test_private_runtime_dispatch_driver_backend_combine_payload_transfer_completion_handoff_scaffold_status_is_backend_owned`.

The handoff scaffold/status consumes backend request scaffold/status input,
dispatch request scaffold/status dependency, combine request scaffold/status
dependency, combine payload descriptor scaffold/status dependency, combine
payload transfer scaffold/status dependency, and combine payload transfer
completion scaffold/status dependency.

The valid prepared handoff scaffold/status remains `unsupported` and keeps
`actual_fused_cross_gpu_execution == 0`. Validation fails closed for owner,
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
as the existing combine-payload scaffold aggregate failure bit.

Red/green evidence: focused red check failed first with
`1 failed in 0.51s` because the handoff scaffold/status API was missing;
focused green check passed with `1 passed in 0.44s` after adding the private
handoff scaffold/status implementation.

Final verification passed before PR creation: focused green passed;
`git diff --check` passed with no output; targeted `markdownlint-cli2`
reported `Summary: 0 error(s)`; the NVIDIA review guard reported
`nvidia review guard passed`; the full private runtime-fusion pytest reported
`25 passed`; and the NVIDIA review-artifact pytest reported `89 passed`.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no handoff implementation, no transport/backend execution,
no scheduler/runtime pass evidence, no fresh H200 fused success, no public
API expansion, no public `TaskArgs`, no public `CallConfig`, no common
runtime C API, no UCCL host-runtime ABI, and no
examples/stable docs/serving/vLLM/DeepSeek/performance claims. It does not
report `persistent_device_uccl_ep_runtime_fusion.status: passed` and does
not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-map`.
This is selected exactly one next PR-sized docs/test map slice for the future
private completion handoff result boundary. It is not a real payload
transfer, completion, handoff, transport, or backend implementation.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Result Map Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-map`
slice after merged PR #203 (`15a66e7b`). This is docs/test mapping only, not
source behavior, for the future private completion handoff result boundary.

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

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-scaffold-status`
slice after merged PR #204 (`f425090f`). This is private completion handoff
result scaffold/status only, not a real result transport/backend
implementation.

The implementation depends on backend request scaffold/status input, the
dispatch request scaffold/status dependency, the combine request
scaffold/status dependency, the combine payload descriptor scaffold/status
dependency, the combine payload transfer scaffold/status dependency, the
combine payload transfer completion scaffold/status dependency, and the
combine payload transfer completion handoff scaffold/status dependency.

Validation records result owner, same invocation, descriptor token,
rank/device, status sink, result sink, public/provenance sourced state, and
fabricated pass evidence. valid prepared result scaffold/status remains
`unsupported`, keeps `actual_fused_cross_gpu_execution == 0`, and records no
actual fused execution claim.

Result status vocabulary includes
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
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. The slice
reuses
`PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
for result scaffold/status validation.

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
private result transport boundary.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Result Transport Map Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-map`
branch/slice after merged PR #205 (`d80ccd23`). This is docs/test mapping
only for the future private result transport boundary, not source behavior.

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

Result transport status vocabulary includes
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

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Result Transport Scaffold Status Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-scaffold-status`
slice after merged PR #206 (`93713fee`). This is private result transport
scaffold/status only, not result transport/backend execution.

The implementation depends on backend request scaffold/status input,
dispatch request scaffold/status dependency, combine request scaffold/status
dependency, combine payload descriptor scaffold/status dependency, combine
payload transfer scaffold/status dependency, combine payload transfer
completion scaffold/status dependency, combine payload transfer completion
handoff scaffold/status dependency, and handoff result scaffold/status
dependency.

Validation records transport owner, same invocation, descriptor token,
rank/device/world size, status sink, result sink, transport sink/handle,
public/provenance sourced state, and fabricated pass evidence. valid
prepared result transport scaffold/status remains `unsupported`, keeps
`actual_fused_cross_gpu_execution == 0`, and records no actual fused
execution claim.

Result transport status vocabulary includes
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

Failure-bit design note: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U` and no public ABI widening. The slice
reuses `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_COMBINE_PAYLOAD_SCAFFOLD`
for result transport scaffold/status validation.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no handoff implementation, no real result transport
implementation, no transport/backend execution, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public API expansion, no public
`TaskArgs`, no public `CallConfig`, no common runtime C API, no UCCL
host-runtime ABI, and no examples/stable docs/serving/vLLM/DeepSeek/performance claims.
It does not report `persistent_device_uccl_ep_runtime_fusion.status: passed`
and does not set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-completion-map`.
This is selected exactly one next PR-sized docs/test map slice for the
future private result transport completion boundary.

## Runtime Dispatch Driver Backend Combine Payload Transfer Completion Handoff Result Transport Completion Map Slice

This selection note records the
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-completion-map`
branch/slice after merged PR #207 (`f91dc06a`). This is docs/test mapping
only for the future private result transport completion boundary, not source
behavior.

PR #207 is accepted as private result transport scaffold/status. This map
depends on backend request scaffold/status input, dispatch request
scaffold/status dependency, combine request scaffold/status dependency,
combine payload descriptor scaffold/status dependency, combine payload
transfer scaffold/status dependency, combine payload transfer completion
scaffold/status dependency, combine payload transfer completion handoff
scaffold/status dependency, handoff result scaffold/status dependency, and
result transport scaffold/status dependency.

The future private result transport completion boundary remains private to
the same invocation. Likely validation concepts are review vocabulary only,
not implementation claims: completion owner, same invocation, descriptor
token, rank/device/world size, status sink, result sink, transport
sink/handle, completion sink/handle, no public/provenance sourced state, and
no fabricated pass evidence.

Result transport completion status vocabulary includes
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_pending`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_unimplemented`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_status_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_result_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_transport_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_completion_sink_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_handle_unbound`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_map_unsupported_boundary`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_owner_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_invocation_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_result_transport_scaffold_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_descriptor_token_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_rank_device_world_size_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_status_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_result_sink_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_transport_sink_handle_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_completion_sink_handle_mismatch`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_public_api_sourced_state`,
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_provenance_sourced_state`,
and
`driver_backend_combine_payload_transfer_completion_handoff_result_transport_completion_fabricated_pass_evidence`.

Failure-bit constraint: `PtoCudaRuntimeFusionFailure` is exhausted through
`1U << 31U`; there is no `1U << 32U`. Future implementation must reuse the
existing combine-payload aggregate failure bit or a documented existing bit,
not widen public ABI.

This slice records no real UCCL-EP dispatch/combine work, no descriptor
allocation behavior change, no payload transfer implementation, no completion
implementation, no handoff implementation, no result transport
implementation, no transport completion implementation, no transport/backend
execution, no scheduler/runtime pass evidence, no fresh H200 fused success,
no public API expansion, no public `TaskArgs`, no public `CallConfig`, no
common runtime C API, no UCCL host-runtime ABI, and no
examples/stable docs/serving/vLLM/DeepSeek/performance claims. It does not
report `persistent_device_uccl_ep_runtime_fusion.status: passed` and does not
set `actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-combine-payload-transfer-completion-handoff-result-transport-completion-scaffold-status`.
This is selected exactly one next PR-sized implementation slice for private
result transport completion scaffold/status only.

## Non-Claims

UCCL PTO host-runtime dispatch, RDMA evidence, multi-node evidence,
serving-level communication evidence, and DeepSeek model correctness remain
pending. The UCCL-EP handoff evidence is adapter/probe evidence only.
UCCL adapter execution is limited to opt-in Python-side probes and handoff
gates. PR #155 accepted only a private unsupported runtime scaffold. It did
not claim fresh H200 fused success, report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, or set
`actual_fused_cross_gpu_execution: true`.
