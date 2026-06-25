# Communication Runtime Boundary

This note records the first PTO-side boundary for NVIDIA communication work.
It constrains communication slices so NCCL experiments do not leak
transport-specific objects into user orchestration code. The concrete
implementation is an internal descriptor and runtime registry in
`simpler_setup/cuda_comm.py`; it is not part of the public `simpler` API.

## Boundary Placement

Communication belongs behind the CUDA platform variant and runtime
implementation, not in the public Python task API:

- Public orchestration continues to pass `Callable`, `TaskArgs`, and
  `CallConfig`.
- `ChipWorker` remains the L2 execution leaf that dispatches into the selected
  host runtime.
- The host runtime owns transport setup for a CUDA platform variant.
- The persistent-device runtime may consume communication descriptors, but it
  should not require Python code to name NCCL objects.

This follows the current architecture docs: the L3+ engine schedules opaque
tasks, while L2 converts `TaskArgs` and `CallConfig` into the chip/runtime ABI.

## Public Surface

The public Python API exposes only normal task data and worker configuration:

- Rank identity is derived from Worker-local device ordering.
- Peer group metadata is held as an opaque internal capability.
- Tensor payloads remain `TaskArgs` tensors or private device pointers.
- Execution knobs remain `CallConfig`.
- Transport selection is a runtime/platform implementation detail.

No Python-visible NCCL transport object is added to `TaskArgs`, `CallConfig`,
or user-facing orchestration.

## Internal Helper

`simpler_setup/cuda_comm.py` provides the first runtime-local boundary shape:

- `CudaCommCapability`: opaque capability id, backend name, rank mapping, and
  supported baseline operation vocabulary.
- `CudaCommHostPlan`: host-side plan that mirrors Worker-style `device_ids`,
  preserving worker index to CUDA device id mapping and deriving one local
  launch plan per rank.
- `CudaCommLaunchPlan`: a local-rank view that resolves rank, CUDA device id,
  world size, and registry runtime id from the opaque capability without
  carrying a transport object.
- `CudaCommDeviceDescriptor`: compact fixed-size descriptor derived from a
  launch plan. The packed layout is five little-endian `uint32` values:
  backend code, rank, CUDA device id, world size, and capability CRC32.
- `create_mock_cuda_comm_capability`: helper for deterministic mock rank
  mapping without importing PyTorch or NCCL.
- `MockCudaCommRuntime`: pure-Python operation semantics for `all_reduce`,
  `reduce_scatter`, `all_gather`, and `send_recv`.
- `TorchNcclCudaCommRuntime`: a local-rank `torch.distributed` NCCL adapter
  that owns device selection, process-group setup, process-group teardown, and
  thin operation forwarding for the same vocabulary.
- `CudaCommRuntimeRegistry`: runtime-private cache that acquires, reuses, and
  releases communicator-like state. Mock entries are keyed by the opaque
  capability id; NCCL entries add `/local_rankN` because process-group state is
  local to a rank process.

The helper is intentionally unexported from `simpler`. It gives runtime tests a
stable internal object shape while host-runtime operation dispatch is brought
up behind the worker boundary.

## Descriptor Handoff

`python/simpler/worker.py` derives a private CUDA communication host plan for
L3 CUDA Workers with `device_ids`. That keeps the rank/device mapping adjacent
to the Worker fork boundary while preserving `TaskArgs`, `CallConfig`, and
normal callable ids as the public task path. When chip children are forked, the
Worker resolves the local `CudaCommLaunchPlan` for each worker index and
attaches both the launch plan and compact `CudaCommDeviceDescriptor` privately
to the child process's Python `ChipWorker` wrapper.

`src/cuda/platform/include/host/pto_cuda_comm_descriptor_abi.h` is the first
C++ host-runtime consumer for those descriptor bytes. It defines the matching
20-byte `PtoCudaCommDeviceDescriptor` layout, backend codes, and
`pto_cuda_comm_descriptor_from_bytes()` for validating the little-endian
Python-packed data. The CUDA host runtime includes this header and exposes
`configure_cuda_comm_descriptor()` as a narrow optional entry point that parses
and stores the descriptor on its `CudaDeviceRunner`.

The chip-child binding path calls `ChipWorker.configure_cuda_comm_descriptor`
with the descriptor bytes immediately after `ChipWorker.init` and before the
child enters its task loop. `ChipWorker` loads the optional
`configure_cuda_comm_descriptor` C symbol with `dlsym`; non-CUDA runtimes are
unchanged because the call is only made when a CUDA launch plan exists.

## Host Runtime NCCL Path

The CUDA host runtime now uses the stored descriptor for communicator
lifecycle. `create_comm_stream_ctx()` snapshots the configured
`PtoCudaCommDeviceDescriptor` into a CUDA-private stream wrapper, and
`comm_init()` validates the requested rank and world size against the
descriptor before returning an opaque CUDA comm handle.

NCCL descriptor-backed communicator setup dynamically loads `libnccl`, rank 0
writes an `ncclUniqueId` through the existing root-info file exchange, peer
ranks read it, and `comm_init()` creates an `ncclComm_t` owned by the CUDA comm
handle. The operation paths are:

- `comm_all_reduce_f32`
- `comm_reduce_scatter_f32`
- `comm_all_gather_f32`
- `comm_send_recv_f32`

They dispatch NCCL all-reduce, reduce-scatter, all-gather, and grouped
send/receive with float32 payloads through the handle and synchronize the
handle's CUDA stream before returning.

`ChipWorker` wraps those float32 operation entries as internal device-pointer
methods. This keeps the operation vocabulary available to chip children and
worker control messages without introducing Python-visible NCCL transport
objects or widening `TaskArgs` / `CallConfig`.

The worker control plane has a process-safe `CTRL_COMM_OP` message for those
baseline operations. The parent stages one fixed-size operation request in
POSIX shared memory per chip, C++ serializes the mailbox transition through
`WorkerThread::control_comm_op`, and the chip child decodes the request before
calling the cached-base-communicator `ChipWorker.comm_*_f32` method.

The CUDA host runtime exposes `comm_last_error()` so `ChipWorker` can surface
transport initialization failures with backend detail instead of a null
communicator. NCCL loading first honors `PTO_CUDA_NCCL_LIBRARY`, then falls
back to `libnccl.so.2` / `libnccl.so`. The worker-control example uses that
contract to discover a venv-bundled `nvidia/nccl/lib/libnccl.so.2` before
forking chip workers.

## Baseline Evidence

`examples/cuda/nccl_two_gpu_baseline.py` builds a `CudaCommHostPlan`, acquires
the NCCL runtime through `CudaCommRuntimeRegistry`, and emits this shape for
the H200 NCCL baseline:

```text
capability_id: nccl:rank0->cuda0,rank1->cuda1
rank_to_device: {"0": 0, "1": 1}
runtime_id: nccl:rank0->cuda0,rank1->cuda1/local_rank0
operations: [all_reduce, reduce_scatter, all_gather, send_recv]
```

The compatibility floor is recorded in
`docs/in_progress/nvidia_backend/nccl_two_h200_baseline.md`.

The worker-control H200 evidence is recorded in
`docs/in_progress/nvidia_backend/nccl_worker_control_h200.md`. The command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-cuda-comm-nccl-worker-control \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    pip install --no-build-isolation -e . >/tmp/pto-cu-pip-install.log && \
    NCCL_DEBUG=WARN .venv/bin/python \
      examples/cuda/nccl_worker_control_ops.py \
      --device-ids 6,7 --tensor-numel 1024 --build --require-cuda'
```

passed on `NVIDIA H200 NVL` devices `6,7` with `tensor_numel: 1024`. The JSON
reported `transport: worker_control`, `status: passed`, and `max_abs_error:
0.0` for `all_reduce`, `reduce_scatter`, `all_gather`, and `send_recv`.

The first communication-coupled persistent MoE handoff gate is recorded in
`docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`.
`examples/cuda/persistent_moe_dispatch_combine.py --with-nccl-handoff` now
runs the existing two-device persistent MoE aggregate, runs the
descriptor-backed NCCL worker-control operations on the same device ids, and
emits `handoff_validation` fields tying `same_device_ids`, persistent MoE
validation, NCCL operation validation, source digests, and bridge digest
matching into one review-safe JSON result.

The first UCCL-EP adapter handoff gate is also recorded in
`docs/in_progress/nvidia_backend/persistent_moe_dispatch_combine_h200.md`.
`examples/cuda/persistent_moe_dispatch_combine.py --with-uccl-ep-handoff`
runs the same two-device persistent MoE aggregate, launches the existing
Python-side UCCL-EP dispatch/combine adapter on the same device ids, and emits
`handoff_validation` fields tying `same_device_ids`, persistent MoE pass
state, UCCL-EP pass state, descriptor metadata, source digests, bridge digest
matching, and zero max errors into one review-safe JSON result. This path is
separate from the NCCL host-runtime worker-control path and does not add CUDA
host-runtime UCCL dispatch.

The reduced fused cross-GPU expert-parallel MoE boundary is represented by
`examples/cuda/persistent_moe_dispatch_combine.py
--with-uccl-ep-fused-boundary`. That mode deliberately records a structured
unsupported boundary when the handoff path passes, because the current runtime
still lacks `persistent_device_uccl_ep_runtime_fusion`: a shared
persistent-device/UCCL-EP boundary that can route dispatch/combine payloads
inside one fused cross-GPU expert-parallel MoE execution. The result is not
fused evidence; it is a review-safe non-evidence marker for the missing
runtime boundary. It is a structured unsupported boundary.
It is not fused evidence.

The current handoff and fused-boundary result shapes now include
`payload_provenance` as dependency evidence. The UCCL-EP adapter side records
only fields emitted by the adapter result: capability id, world size,
rank-to-CUDA-device mapping, descriptor dimensions, descriptor metadata
shapes, and rank results. The persistent-device graph side records only
existing graph inputs and outputs: graph descriptor id, device ids,
rank/device mapping, source digests, and bridge digest.

This provenance does not create shared payload ownership. The result records
no shared ownership token, `shared_payload_ownership.exists: false`, and an
empty lifetime transition log. The
`persistent_device_uccl_ep_runtime_fusion.status` field remains `unsupported`
until a runtime component creates and transfers a real cross-component payload
boundary.

The blocked implementation slice adds local guard code in
`examples/cuda/persistent_moe_dispatch_combine.py` rather than changing that
status. `_validate_runtime_fusion_evidence` accepts only evidence whose
producer is `persistent_device_uccl_ep_runtime_fusion`, whose descriptor is
marked runtime-owned, whose rank/device mapping matches the Worker-local
device ordering, and whose ownership token and lifetime transition log are
complete. Pass-like fields copied from UCCL-EP adapter output, payload
provenance, or handoff metadata are rejected with
`fabricated_or_untrusted_pass_evidence` and cannot set
`actual_fused_cross_gpu_execution: true`.

## UCCL-EP Runtime Fusion Contract

This design/dependency slice names the missing runtime boundary. It does not
implement `persistent_device_uccl_ep_runtime_fusion`.

`persistent_device_uccl_ep_runtime_fusion` is the internal contract between a
persistent-device graph descriptor and an opt-in UCCL-EP dispatch/combine
runtime. Its implementation must sit behind the CUDA runtime boundary, not in
`TaskArgs`, `CallConfig`, or user orchestration code.

The CUDA persistent-device runtime run context may hold runtime-owned shared
payload descriptors. The host side of that context is reachable only through
the existing `ChipWorker` to host-runtime `.so` edge, and it may allocate a
device-visible descriptor buffer before launching the persistent-device
scheduler. Python may pass the existing callable id, task payloads,
`CallConfig`, device ids, and private communication launch metadata, but it
must not manufacture the shared descriptor, ownership token, or lifetime log.

The component that records the ownership token and payload lifetime transition
log is the persistent-device/UCCL-EP runtime fusion coordinator. It is a
private runtime component inside the `persistent_device` CUDA runtime, scoped
to one `ChipWorker::run` invocation. The persistent-device graph can publish
graph provenance and the UCCL-EP runtime can publish adapter provenance, but
neither component may independently synthesize a shared ownership token. The
fusion coordinator issues the token, validates every state transition, and
emits the transition log in the fused-boundary result.

PR #147 provenance is accepted input evidence only. It proves that the
UCCL-EP adapter and persistent-device graph can each report descriptor and
rank provenance. It does not prove that a runtime-owned shared payload
descriptor exists. Until the fusion coordinator creates the shared descriptor,
`persistent_device_uccl_ep_runtime_fusion.status` remains `unsupported`,
`actual_fused_cross_gpu_execution` remains `false`, and no shared ownership
token or payload lifetime transition log exists.

The contract must expose these reviewable fields before an implementation can
claim actual fused cross-GPU expert-parallel MoE execution:

- graph descriptor id and persistent-device runtime id;
- UCCL capability id and rank-to-CUDA-device mapping;
- dispatch payload descriptor, including token count, hidden size, top-k,
  expert count, input dtype, and metadata shape summary;
- combine payload descriptor using the same ownership token as dispatch;
- payload owner field with allowed values `persistent_device_graph`,
  `uccl_ep_runtime`, and `released`;
- payload lifetime state with allowed values `allocated`, `dispatch_ready`,
  `dispatch_in_flight`, `combine_ready`, `combine_in_flight`, `complete`, and
  `released`;
- status fields for `persistent_moe`, `uccl_ep_runtime`,
  `persistent_device_uccl_ep_runtime_fusion`, and
  `actual_fused_cross_gpu_execution`;
- failure fields for setup, descriptor, rank/device, payload lifetime,
  transport, validation, and unsupported-boundary failures.

Dispatch and combine payload ownership must transfer exactly once per phase.
The persistent-device graph owns task buffers until it publishes a
`dispatch_ready` payload descriptor. The UCCL-EP runtime owns the descriptor
and payload views while dispatch or combine is in flight. Ownership returns to
the persistent-device graph only after UCCL-EP records `combine_ready` with a
matching ownership token. The implementation must record mismatched tokens,
double release, use-after-release, and leaked in-flight ownership as failures,
not as skips.

Rank/device mapping is derived from the same Worker-local device ordering used
by the existing communication boundary. The evidence shape must report
`rank_to_device`, `device_ids`, `world_size`, and the UCCL capability id. A
rank/device mismatch between the persistent graph and UCCL-EP runtime is a
failure, even when either component can pass independently.

Mandatory failure states include descriptor shape mismatch, missing ownership
token, mismatched ownership token, illegal lifetime transition, double release,
use-after-release, leaked in-flight ownership, rank/device mismatch, UCCL
transport failure, persistent-device scheduler failure, numeric validation
failure, and unsupported runtime boundary. These states must be reported as
`failed`, `setup_failed`, or `unsupported`; they must not be downgraded to
skips.

The current local guard reports these states in `failure_fields`. The normal
fused-boundary result is still `unsupported` with
`failure_fields.unsupported_boundary: persistent_device_uccl_ep_runtime_fusion`
because no trusted runtime-owned descriptor source exists. If a handoff or
adapter result tries to supply pass-like runtime-fusion fields, the guard marks
the fused-boundary result `failed` and records the rejection in
`failure_fields`; it does not treat that metadata as runtime-owned evidence.

Status values must stay review-safe:

- `passed`: setup completed, persistent MoE passed, UCCL-EP runtime passed,
  payload ownership and lifetime checks passed, rank/device mapping matched,
  and `actual_fused_cross_gpu_execution` is `true`.
- `unsupported`: prerequisites completed far enough to identify the missing
  `persistent_device_uccl_ep_runtime_fusion` implementation or another named
  unsupported runtime boundary.
- `setup_failed`: dependency import, build, extension load, GPU discovery,
  process-group launch, or remote environment setup failed before the boundary
  contract could run.
- `failed`: the boundary ran and produced a validation, descriptor,
  rank/device, payload lifetime, transport, or numeric correctness failure.

Unsupported and setup-failed states are non-evidence for fused execution.
They may be useful review evidence for dependency shape only when they include
the named failing boundary, command, device ids, dependency placeholders, and
non-claims.

Non-evidence states also include independent two-device persistent MoE
baselines, Python-side UCCL-EP adapter passes, NCCL worker-control passes,
payload provenance without a shared ownership token, stale H200 artifacts, and
any result where `actual_fused_cross_gpu_execution` is `false`. These states
can remain useful dependency evidence, but they must not be described as
actual fused cross-GPU expert-parallel MoE execution.

## Runtime Fusion Coordinator Boundary

The missing owner is a CUDA persistent-device runtime component:
`persistent_device_uccl_ep_runtime_fusion`. It is scoped to one
`ChipWorker::run` invocation and sits below the Python example handoff. The
example may request the fused-boundary mode and carry adapter or graph
provenance, but it cannot allocate the shared dispatch/combine descriptor,
issue the ownership token, or claim payload lifetime transitions. Those
fields are only trustworthy when emitted behind the `ChipWorker` to CUDA host
runtime edge.

The reviewable entry point shape is:

1. `WorkerThread` dispatches a normal chip task with `Callable`, `TaskArgs`,
   and `CallConfig`.
2. The chip child decodes the mailbox payload and calls `ChipWorker::run`.
3. `ChipWorker::run` converts the decoded view to `ChipStorageTaskArgs` and
   calls the CUDA host-runtime entry point for the selected callable id.
4. The CUDA persistent-device runtime run context detects the opt-in
   UCCL-EP fusion capability and constructs
   `persistent_device_uccl_ep_runtime_fusion`.
5. The coordinator allocates one host-side control record plus one
   device-visible dispatch/combine descriptor buffer before launching the
   persistent-device scheduler.
6. The coordinator passes descriptor views to the persistent-device graph and
   UCCL-EP runtime adapter without exposing those views through public
   `TaskArgs`, `CallConfig`, or example-owned JSON fields.
7. The coordinator collects status, validation, failure fields, the ownership
   token, and the lifetime transition log after scheduler and UCCL-EP work
   complete.

The descriptor allocation site is the CUDA persistent-device runtime run
context, not `examples/cuda/persistent_moe_dispatch_combine.py`. The owner of
the allocation is the coordinator. The persistent-device graph may fill graph
descriptor provenance and scheduler validation. The UCCL-EP runtime may fill
transport validation and dispatch/combine adapter provenance. Neither side may
independently mark the shared descriptor as runtime-owned.

The ownership token issuer is also the coordinator. A later pass result must
show exactly one token shared by dispatch and combine descriptor records. The
token must be created before `dispatch_ready`, must be present on every
transition, and must be released only after `complete`.

The required state machine is:

```text
allocated
  -> dispatch_ready
  -> dispatch_in_flight
  -> combine_ready
  -> combine_in_flight
  -> complete
  -> released
```

Every transition records actor, state, token, descriptor id, rank/device map,
and result status. Legal actors are `persistent_device_graph`,
`uccl_ep_runtime`, and
`persistent_device_uccl_ep_runtime_fusion`. Illegal transition, missing token,
mismatched token, double release, use-after-release, leaked in-flight owner,
or missing release is a payload lifetime failure.

Failure-field responsibilities are split so review can identify the failing
owner:

- `setup`: dependency import, CUDA device discovery, build, extension load, or
  process launch before the boundary can run.
- `unsupported_boundary`: missing coordinator, missing UCCL-EP runtime path,
  or missing persistent-device fusion capability.
- `descriptor`: shape, dtype, metadata, or descriptor allocation mismatch.
- `rank_device`: mismatch between Worker-local device ordering, graph
  rank/device mapping, and UCCL capability mapping.
- `payload_lifetime`: token, owner, transition, release, or leak error.
- `transport`: UCCL-EP dispatch/combine runtime failure after setup.
- `scheduler`: persistent-device scheduler error, incomplete task count, or
  nonzero fan-in remaining.
- `validation`: numeric validation failure or missing required pass field.
- `fabricated_or_untrusted_pass_evidence`: pass-like fields supplied by
  adapter output, payload provenance, handoff metadata, or example-side JSON
  instead of the coordinator.

Local tests required before implementation may report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true` must exercise this entry point shape
and failure split. They must reject pass results without a runtime-owned
descriptor allocation site, one coordinator-issued ownership token, a complete
transition log, matching rank/device maps, and empty failure fields. Future
H200 evidence must use the existing
`--with-uccl-ep-fused-boundary` command shape and may report passed/true only
when the fresh JSON result is produced by the runtime coordinator rather than
by example-side handoff metadata.

## Private Runtime Entry Contract

The private callable entry is owned by the CUDA persistent-device runtime. The
contract name is `persistent_device_uccl_ep_runtime_fusion`, and the
host-callable path is the internal
`persistent_device_uccl_ep_runtime_fusion_entry` reached from
`ChipWorker::run` after `ChipStorageTaskArgs` has been assembled. This entry
is not a public UCCL host-runtime ABI and is not exposed through `simpler`.

`ChipWorker::run` and `ChipStorageTaskArgs` request coordinator construction
only by carrying existing private runtime state into the CUDA host-runtime
callable path:

- the normal callable id selected by the mailbox dispatch;
- the `ChipStorageTaskArgs` tensor/scalar view produced from `TaskArgs`;
- the unchanged `CallConfig`;
- the chip-local rank/device map that the Worker already derived from
  private CUDA communication launch metadata;
- an internal persistent graph descriptor handle associated with the callable;
- an opaque UCCL-EP capability metadata handle attached to the chip child;
- private descriptor allocation and validation policies selected by the CUDA
  persistent-device runtime.

No field is added to public `TaskArgs` or public `CallConfig`. No user-visible
Python object can request a pass result directly.

The coordinator construction request has these minimum fields:

- `callable_id`: the mailbox callable id for the prepared persistent graph;
- `rank_device_map`: chip-local rank, CUDA device id, world size, and
  Worker-local device ordering;
- `persistent_graph_descriptor`: an internal handle for the persistent MoE
  graph descriptor and its graph provenance;
- `uccl_ep_capability`: opaque capability id, world size, transport mode,
  descriptor vocabulary, and adapter provenance handles;
- `descriptor_allocation_policy`: runtime-owned allocation requirement,
  host-control record policy, device-visible descriptor buffer policy, and
  dispatch/combine token-sharing rule;
- `validation_policy`: required rank/device match, descriptor shape and dtype
  checks, lifetime transition checks, scheduler checks, transport checks, and
  numeric validation checks;
- `output_sink`: the runtime-owned status artifact writer for
  `persistent_device_uccl_ep_runtime_fusion`.

The coordinator result returned to the host/runtime status artifact has these
minimum fields:

- `coordinator_status`: `passed`, `unsupported`, `setup_failed`, or `failed`;
- `descriptor_allocation_provenance`: allocator owner, host-control record
  id, device-visible descriptor id, dispatch descriptor id, combine descriptor
  id, and runtime-owned allocation flag;
- `ownership_token`: one coordinator-issued token shared by dispatch and
  combine descriptors, or `null` when unsupported or setup failed;
- `state_transitions`: ordered transition records with actor, state, token,
  descriptor id, rank/device map, and status;
- `rank_device_map`: the Worker-local device ordering and UCCL capability map
  used by the coordinator;
- `validation_summary`: scheduler, transport, descriptor, lifetime, rank
  device, and numeric validation outcomes;
- `failure_fields`: explicit setup, unsupported, descriptor, rank/device,
  payload lifetime, transport, scheduler, validation, and
  fabricated/untrusted evidence failures.

Forbidden data paths remain non-evidence. Example-side JSON, adapter-only
provenance, handoff metadata, public `TaskArgs`, and public `CallConfig` must
not synthesize `persistent_device_uccl_ep_runtime_fusion.status: passed`,
`actual_fused_cross_gpu_execution: true`, an ownership token, allocation
provenance, or transition log. If any of those paths supplies pass-like
fields, the result must be `failed` with
`failure_fields.fabricated_or_untrusted_pass_evidence`.

Failure behavior stays explicit:

- `unsupported`: the private entry, UCCL-EP capability, persistent graph
  descriptor, descriptor allocation policy, or validation policy is absent.
- `setup_failed`: dependency import, CUDA setup, extension load, build,
  process launch, or private metadata setup fails before the coordinator can
  validate the request.
- `failed`: descriptor allocation, rank/device agreement, payload lifetime,
  UCCL-EP transport, persistent-device scheduler, validation, or fabricated
  pass-evidence checks fail after the coordinator boundary is reached.

The entry contract is a dependency boundary only. It names the private request
and result fields that a later implementation must satisfy before it can
construct a coordinator and emit trusted fused-boundary evidence.

PR #153 accepted this entry contract as private dependency evidence only. It
did not implement CUDA runtime behavior, add UCCL host-runtime ABI fields,
change the fused-boundary result shape, claim fresh H200 fused success, report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, or set
`actual_fused_cross_gpu_execution: true`.

PR #155 accepted the private unsupported entry scaffold as the next narrow
implementation slice. It added private request/result plumbing behind the
CUDA persistent DAG host-runtime path, but the review-safe result remains
`unsupported` because the runtime coordinator still does not create shared
dispatch/combine descriptor ownership, issue the ownership token, record the
complete lifetime transition log, or emit trusted coordinator-owned
validation fields.

That slice now introduces a private CUDA host-side scaffold, not a pass
implementation. `src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h`
defines `PtoCudaRuntimeFusionRequest`,
`PtoCudaRuntimeFusionResult`, and
`persistent_device_uccl_ep_runtime_fusion_entry`. The CUDA host runtime calls
the entry from
`src/cuda/platform/onboard/host/pto_runtime_c_api.cpp` when the persistent DAG
path has a graph descriptor. The request is populated only from private
runtime state currently available there: callable id, persistent graph
descriptor pointer, private CUDA rank/device descriptor when configured, and
a runtime-owned output sink.

The scaffold keeps normal evidence unsupported. Missing coordinator,
descriptor allocator, UCCL-EP runtime path, validation policy,
`ChipStorageTaskArgs`, UCCL-EP capability metadata, or rank/device metadata
sets explicit failure bits and keeps
`actual_fused_cross_gpu_execution` false. Adapter provenance, example-side
JSON, handoff metadata, payload provenance, public `TaskArgs`, and public
`CallConfig` are rejected as `fabricated_or_untrusted_pass_evidence` if they
try to provide pass-like fields.

The coordinator scaffold/status slice narrows only the missing coordinator
state. `PtoCudaRuntimeFusionCoordinator` is private to
`pto_cuda_runtime_fusion_abi.h`, and
`pto_cuda_runtime_fusion_prepare_private_coordinator` binds one invocation id
to the accepted descriptor allocation, coordinator-owned UCCL-EP runtime path,
unsupported/failure status, and runtime-owned output sink. The CUDA host
runtime stores this as `runtime_fusion_coordinator_` from
`CudaDeviceRunner::record_runtime_fusion_unsupported`.

This clears `missing_coordinator` only when the request fields point at the
coordinator-owned descriptor allocation and runtime path. The final result is
still `unsupported`; UCCL-EP runtime dispatch is not implemented, no pass
evidence is emitted, and `actual_fused_cross_gpu_execution` remains false.

PR #157 (`nvidia-uccl-ep-runtime-fusion-chip-storage-task-args-request`) is
closed invalid. It tried to satisfy the missing `ChipStorageTaskArgs` field by
assigning the persistent DAG run `args` pointer to
`PtoCudaRuntimeFusionRequest::chip_storage_task_args` and recording
`sizeof(ChipStorageTaskArgs)`. At that call site the pointer is a
`PtoCudaPersistentDagArgs *`, not a `ChipStorageTaskArgs *`, so the branch
fabricated the private request boundary and remains a blocked handoff only.

The `nvidia-uccl-ep-runtime-fusion-private-request-envelope` slice narrows that
handoff. `pto_cuda_private_run_envelope.h` carries runtime-specific task args
separately from a typed `ChipStorageTaskArgs` pointer, and
`PtoCudaRuntimeFusionRequest::chip_storage_task_args` is now a
`const ChipStorageTaskArgs *`. The CUDA host runtime keeps
`PtoCudaPersistentDagArgs *` as the persistent DAG launch input and copies only
`envelope->chip_storage_task_args` into the runtime-fusion request when a valid
private envelope is supplied. `ChipWorker::run` probes the optional CUDA-only
`run_prepared_with_cuda_private_args` symbol, but explicitly rejects the
typed-args private-envelope path because it cannot supply the required
runtime-specific `PtoCudaPersistentDagArgs *`.

This does not add public `TaskArgs`, public `CallConfig`, common runtime C API
fields, UCCL host-runtime ABI fields, pass evidence, RDMA, multi-node,
serving, vLLM, DeepSeek, throughput, or latency claims. Missing coordinator,
descriptor allocator, UCCL-EP runtime path, validation policy, UCCL-EP
capability metadata, or pass evidence still keeps the result unsupported or
failed.

## Runtime Args Handoff Map

The next private dependency is the association boundary between the typed
chip-storage task args and the runtime-specific persistent DAG args. The
selected map preserves PR #160's separation: the two pointers are separate
fields in `PtoCudaPrivateRunArgsEnvelope`, and each field is valid only when
it points at a real object created by its owning layer.

`ChipWorker::run` remains the only owner of the `ChipStorageTaskArgs` object
assembled from the mailbox `TaskArgs` view. It may hand the address of that
object to a CUDA-private hook for the duration of the call, but it must not
reinterpret that object as persistent DAG runtime args. The private hook now
carries only the typed chip-storage pointer, its expected size, callable id,
and a per-`ChipWorker` invocation id into the CUDA host runtime. It still does
not set `runtime_task_args` from the chip-storage pointer.

The CUDA persistent DAG host-runtime path is the owner of
`PtoCudaPersistentDagArgs *`. That path resolves the prepared persistent DAG
callable, validates the persistent DAG state, and constructs the runtime
launch request. Only there can the implementation populate
`PtoCudaPrivateRunArgsEnvelope::runtime_task_args` with the
`PtoCudaPersistentDagArgs *` used by that persistent DAG invocation while also
carrying the
`chip_storage_task_args` pointer received from the same `ChipWorker::run`
invocation.

The reviewable private association point is therefore:

```text
ChipWorker::run
  -> const ChipStorageTaskArgs *
  -> CUDA private host-runtime handoff
  -> prepared persistent DAG callable lookup
  -> PtoCudaPersistentDagArgs *
  -> PtoCudaPrivateRunArgsEnvelope
  -> persistent_device_uccl_ep_runtime_fusion_entry
```

The association is valid only when both pointers are same-invocation inputs,
the envelope records `sizeof(ChipStorageTaskArgs)` for the chip-storage
field, and the runtime-task-args size matches the CUDA persistent DAG args
type used by the prepared callable. Null, stale, wrong-size, wrong-callable,
or cross-invocation envelopes are implementation failures, not skips.

The private host-runtime handoff implementation now adds that association
plumbing and local coverage for null pointers, wrong sizes, mismatched
callable types, stale envelopes, cross-invocation envelopes, and forbidden
public/API evidence paths. The fused-boundary status remains unsupported or
failed and must not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

PR #162 accepted this map as docs/test dependency evidence only. PR #164 then
accepted
`nvidia-uccl-ep-runtime-fusion-private-host-runtime-handoff`: a narrow
private CUDA persistent DAG host-runtime handoff that associates only real
same-invocation `ChipStorageTaskArgs *` and `PtoCudaPersistentDagArgs *`
pointers. It did not expand public `TaskArgs`, public `CallConfig`, the
common runtime C API, or UCCL host-runtime ABI fields.

PR #164 implements that handoff and keeps the coordinator, descriptor
allocator, UCCL-EP runtime path, validation policy, UCCL-EP capability
metadata, pass evidence, and H200 fused-success evidence absent. PR #165 then
recorded a docs/test status refresh and selected the capability metadata map.
PR #166 accepted only the private UCCL-EP capability metadata map; it did not
implement validation policy or CUDA runtime behavior.

## Capability Metadata Map Slice

This branch maps private UCCL-EP capability metadata only. The metadata stays
inside the CUDA persistent-device runtime path and chip-child private metadata;
it is not promoted into public `TaskArgs`, public `CallConfig`, the common
runtime C API, or UCCL host-runtime ABI fields.

The later `persistent_device_uccl_ep_runtime_fusion_entry` coordinator request
needs only this private metadata vocabulary:

- capability id;
- world size;
- rank-to-device map;
- descriptor vocabulary for dispatch/combine payload metadata;
- transport mode;
- adapter provenance handles;
- setup/validation failure ownership.

The PR #164 association between real same-invocation `ChipStorageTaskArgs *`
and `PtoCudaPersistentDagArgs *` remains the only accepted request-args
handoff. Capability metadata is an additional private dependency, not a way to
replace either pointer or synthesize runtime-fusion evidence.

The cases missing, stale, mismatched-rank, mismatched-world-size, or
public/API-sourced capability metadata must report `unsupported` or `failed`.
Missing private metadata is an unsupported prerequisite; stale metadata, rank
mismatch, world size mismatch, and public/API-sourced capability metadata are
validation failures when a boundary attempts to use them.

The forbidden pass-evidence paths remain explicit: public `TaskArgs`, public
`CallConfig`, common runtime C API, UCCL host-runtime ABI, example JSON,
adapter provenance, and handoff metadata must not supply fields that set
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

This is a docs/test dependency map with no CUDA runtime behavior change. It
has no runtime-fusion coordinator implementation, no descriptor allocator
implementation, no UCCL-EP runtime path implementation, no validation policy
implementation, and no fresh H200 fused-success evidence. PR #166 merged this
scope as `42b996666e279024b43f490a310c490a591a897d`.

## Validation Policy Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-validation-policy-map`.

PR #168 accepted only the private validation policy dependency map required
after PR #166's capability metadata vocabulary and before descriptor
allocation, UCCL-EP runtime dispatch, coordinator implementation, pass
evidence, or H200 fused-success evidence.

The validation policy remains private to the CUDA persistent-device runtime
path. It validates PR #164 same-invocation request args and PR #166 capability
metadata together, including capability id, world size, rank-to-device map,
descriptor vocabulary, transport mode, adapter provenance handles, and
setup/validation failure ownership.

Failure ownership is explicit:

- missing metadata is unsupported;
- stale metadata is failed;
- mismatched-rank metadata is failed;
- mismatched-world-size metadata is failed;
- descriptor-vocabulary mismatch is failed because descriptor vocabulary must
  match dispatch/combine payload terms;
- transport-mode mismatch is failed because transport mode must be `ep`;
- adapter-provenance mismatch is failed because adapter provenance handles
  must match the private capability id, invocation id, and rank/device map;
- public/API-sourced metadata is failed as fabricated or untrusted pass
  evidence.

This slice has no descriptor allocation policy implementation, no UCCL-EP
runtime dispatch, no coordinator implementation, no pass evidence, and no
H200 fused-success evidence. Public `TaskArgs`, public `CallConfig`, common
runtime C API fields, UCCL host-runtime ABI fields, example JSON, adapter
provenance, and handoff metadata remain forbidden pass-evidence paths.

PR #168 merged this validation policy scope as
`e33d232deccdf947b9c382a3605191d0d5ae0004`.

## Descriptor Allocation Policy Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-policy-map`.

This dependency slice maps only the private descriptor allocation policy
required after PR #168's validation policy and before UCCL-EP runtime
dispatch, coordinator implementation, pass evidence, or H200 fused-success
evidence. It does not implement descriptor allocation.

The descriptor allocation policy remains private to the CUDA
persistent-device runtime path. It preserves PR #164 same-invocation request
args, PR #166 UCCL-EP capability metadata, and PR #168 validation policy as
prerequisites rather than pass evidence.

The allocator owner is the future private
`persistent_device_uccl_ep_runtime_fusion` coordinator inside one CUDA
persistent-device runtime run context. The host-control record policy is private
per invocation: the record may identify invocation id, persistent graph
descriptor id, UCCL capability id, validated rank/device map, descriptor
vocabulary, allocation state, runtime owner, and shared ownership token slot,
but those fields are not public API or ABI fields.

The device-visible descriptor buffer policy is also private: buffers are future
coordinator-owned allocations from the CUDA persistent-device runtime
allocator, visible only to the persistent-device scheduler and UCCL-EP
runtime path. The dispatch descriptor identity is the validated graph
descriptor id, capability id, invocation id, rank/device map, dispatch
vocabulary, payload shape, and coordinator-issued shared token. The combine
descriptor identity uses the same validated ids, rank/device map, combine
vocabulary, payload shape, and the same shared token as dispatch.

Failure ownership is explicit: missing policy is unsupported, stale policy is
failed, non-runtime-owned allocation is failed, descriptor-vocabulary mismatch
is failed, token-sharing mismatch is failed, rank/device mismatch is failed,
and public/API-sourced policy fields are failed as fabricated or untrusted
pass evidence. The shared-token requirement is strict because dispatch and
combine descriptor identities must share exactly one coordinator-issued
ownership token. The allocation lifetime failure ownership belongs to that
same private runtime owner.

This slice must not implement UCCL-EP runtime dispatch, construct the
coordinator, change CUDA runtime behavior, allocate descriptors, claim pass
evidence, or claim H200 fused-success evidence. Public `TaskArgs`, public
`CallConfig`, common runtime C API fields, UCCL host-runtime ABI fields,
example JSON, adapter provenance, and handoff metadata remain forbidden
pass-evidence paths.

PR #170 accepted only the private descriptor allocation policy scope as
`bd0b59ee8d5afc969020d3aea047aafc9f3152be`: allocator owner, host-control
record policy, device-visible descriptor buffer policy, dispatch/combine
descriptor identity, shared-token requirement, and allocation lifetime failure
ownership. It did not implement CUDA runtime behavior, descriptor allocation,
UCCL-EP runtime dispatch, a coordinator, pass evidence, or H200 fused-success
evidence.

## UCCL-EP Runtime Path Map Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-map`.

This accepted dependency slice maps the private UCCL-EP runtime path after
PR #170
descriptor allocation policy. It is a map only: it defines how a later
runtime-fusion coordinator would pass coordinator-owned dispatch and combine
descriptor views into UCCL-EP runtime logic, but it must not implement that
runtime path.

The runtime path remains private to the CUDA persistent-device runtime path.
It consumes PR #164 same-invocation request args, PR #166 capability metadata,
PR #168 validation policy, and PR #170 descriptor allocation policy as
prerequisites. Those inputs remain prerequisites rather than pass evidence.

The map must define the runtime-path owner, dispatch descriptor handoff,
combine descriptor handoff, descriptor-token checks, rank/device checks,
transport-mode checks, and runtime-path failure ownership. The runtime-path
owner is the future private `persistent_device_uccl_ep_runtime_fusion`
coordinator inside one CUDA persistent-device runtime run context.

The dispatch descriptor handoff may consume only the PR #170 dispatch
descriptor identity: invocation id, persistent graph descriptor id, UCCL
capability id, validated rank/device map, descriptor vocabulary, dispatch
payload shape, and coordinator-issued shared token. The combine descriptor
handoff may consume only the matching PR #170 combine descriptor identity with
the same invocation id, persistent graph descriptor id, UCCL capability id,
validated rank/device map, descriptor vocabulary, combine payload shape, and
exactly the same coordinator-issued shared token.

descriptor-token checks fail unless dispatch and combine descriptor views
carry the same coordinator-issued token and that token belongs to the current
same-invocation request. Rank/device checks fail unless the persistent graph
descriptor, private UCCL-EP capability metadata, and Worker-local CUDA device
ordering agree. Transport-mode checks fail unless the private UCCL-EP
capability metadata declares `transport mode: ep` before either descriptor
handoff is consumed.

Runtime-path failure ownership is private to the future coordinator. missing
runtime path is unsupported. stale descriptor views are failed,
descriptor-token mismatch is failed, rank/device mismatch is failed,
transport-mode mismatch is failed, descriptor-vocabulary mismatch is failed,
and public/API-sourced runtime-path fields are failed as fabricated or
untrusted pass evidence.

This accepted slice must not implement UCCL-EP runtime dispatch, construct a
coordinator, allocate descriptors, change CUDA runtime behavior, claim pass
evidence, or claim H200 fused-success evidence. Public `TaskArgs`, public
`CallConfig`, common runtime C API fields, UCCL host-runtime ABI fields,
example JSON, adapter provenance, and handoff metadata remain forbidden
pass-evidence paths.

PR #172 accepted only this private UCCL-EP runtime path dependency map as
`21b2b32a475dc04e19700115af74510daef70859`. The accepted scope is the
runtime-path owner, dispatch descriptor handoff, combine descriptor handoff,
descriptor-token checks, rank/device checks, transport-mode checks, and
runtime-path failure ownership. It did not implement CUDA runtime behavior,
UCCL-EP runtime dispatch, a coordinator, descriptor allocation, pass
evidence, or H200 fused-success evidence.

## Accepted UCCL-EP Runtime Path Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-uccl-ep-runtime-path-impl`.

This slice adds the narrow private UCCL-EP runtime path implementation
scaffold. It starts from the PR #172 map and adds private runtime-path
plumbing below the CUDA persistent-device runtime boundary, without
constructing the runtime-fusion coordinator or allocating dispatch/combine
descriptor memory.

Required implementation boundaries:

- consume PR #164 same-invocation request args, PR #166 UCCL-EP capability
  metadata, PR #168 validation policy, PR #170 descriptor allocation policy,
  and PR #172 runtime-path map only as prerequisites;
- keep descriptor-token, rank/device, transport-mode, descriptor-vocabulary,
  stale-descriptor, and public/API-sourced runtime-path failures explicit;
- preserve missing descriptor allocation and missing coordinator as
  unsupported or failed states;
- keep public `TaskArgs`, public `CallConfig`, the common runtime C API,
  UCCL host-runtime ABI fields, example JSON, adapter provenance, and handoff
  metadata out of pass-evidence paths.

Implemented private scaffold:

- `PtoCudaUcclEpRuntimePath` and `PtoCudaUcclEpRuntimeDescriptorView` are
  private CUDA runtime-fusion inputs carried through the existing internal
  `uccl_ep_runtime` pointer;
- `PtoCudaRuntimeFusionRequest::invocation_id` connects descriptor views to
  the same private run envelope as `ChipStorageTaskArgs` and records the
  same-invocation id used by runtime-path checks;
- runtime-path validation reports failed stale descriptor views,
  descriptor-token mismatch, rank/device mismatch, transport-mode mismatch,
  descriptor-vocabulary mismatch, and public/API-sourced runtime-path fields;
- valid private runtime-path descriptors still do not produce pass evidence
  while the coordinator and descriptor allocator are absent.

Required non-claims:

- no runtime-fusion coordinator implementation;
- no descriptor allocation implementation;
- no pass evidence;
- no fresh H200 fused-success evidence;
- no `persistent_device_uccl_ep_runtime_fusion.status: passed`;
- no `actual_fused_cross_gpu_execution: true`;
- no RDMA, multi-node, serving, vLLM, DeepSeek, throughput, or latency claim.

PR #174 accepted this private runtime-path scaffold as
`3b4b19a04855d27289fb9cdad802fee0c47d8265`. The accepted surface is limited
to `PtoCudaUcclEpRuntimePath`, `PtoCudaUcclEpRuntimeDescriptorView`, private
descriptor-view validation, and invocation-id propagation through private
CUDA runtime-fusion request state. It did not implement the runtime-fusion
coordinator, descriptor allocation, UCCL-EP runtime dispatch, pass evidence,
fresh H200 fused-success evidence, public `TaskArgs`, public `CallConfig`,
common runtime C API fields, UCCL host-runtime ABI fields, serving, vLLM,
DeepSeek, throughput, or latency evidence.

## Accepted Descriptor Allocation Implementation Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-descriptor-allocation-impl`.

This slice is a private descriptor allocation implementation only. It adds
the host-control record and device-visible dispatch/combine descriptor buffer
mechanics required by the PR #170 allocation policy and binds them to the same
invocation id carried by the PR #174 runtime-path scaffold.

Implementation evidence:

- `PtoCudaUcclEpDescriptorHostControl` records the private per-invocation
  host-control fields: invocation id, persistent graph descriptor, validated
  rank/device map, descriptor vocabulary, allocation state, runtime-owned
  flag, shared token, and descriptor offsets.
- `PtoCudaUcclEpDeviceDescriptorBuffer` models the device-visible dispatch and
  combine descriptor buffer. The dispatch and combine records carry matching
  invocation id, graph descriptor, rank/device map, descriptor vocabulary, and
  shared token.
- `PtoCudaUcclEpDescriptorAllocation` binds the host-control record, the
  device-visible buffer, and the PR #174
  `PtoCudaUcclEpRuntimePath` descriptor views into one private allocation.
- `pto_cuda_runtime_fusion_allocate_uccl_ep_descriptors` builds that
  allocation from a private `PtoCudaRuntimeFusionRequest`, requiring the
  request's same-invocation id, graph descriptor, rank/device metadata, device
  buffer, and nonzero shared token.
- `src/cuda/platform/onboard/host/pto_runtime_c_api.cpp` calls the allocator
  from the persistent DAG private host-runtime path and wires the resulting
  allocation and runtime path into the private request state.

This scope is explicitly narrower than constructing
`persistent_device_uccl_ep_runtime_fusion` as a coordinator and narrower than
dispatching UCCL-EP runtime work. Missing coordinator behavior and missing
UCCL-EP runtime dispatch must remain unsupported or failed states. Public
`TaskArgs`, public `CallConfig`, common runtime C API fields, UCCL
host-runtime ABI fields, example JSON, adapter provenance, handoff metadata,
and payload provenance remain forbidden pass-evidence paths.

This slice must not claim pass evidence, fresh H200 fused-success evidence,
`persistent_device_uccl_ep_runtime_fusion.status: passed`,
`actual_fused_cross_gpu_execution: true`, RDMA, multi-node, serving, vLLM,
DeepSeek, throughput, or latency.

PR #176 accepted this private descriptor allocation scaffold only, merged as
`6e0cecc174ae9db47573c4c0f1698be7accb295c`. The accepted surface is limited
to the private host-control record, device-visible dispatch/combine descriptor
buffer mechanics, allocation bundle, same-invocation binding, and private
runtime-path handoff into request state. It did not implement coordinator
construction, UCCL-EP runtime dispatch, pass evidence, fresh H200
fused-success evidence, public `TaskArgs`, public `CallConfig`, common
runtime C API fields, UCCL host-runtime ABI fields, examples, stable docs,
serving, vLLM, DeepSeek, throughput, or latency.

## Accepted Runtime Fusion Coordinator Scaffold Status Slice

PR #178 merged as `aea89cc9dea8560602c72f84e5ff6e78ca526434` and accepted
only the private coordinator scaffold/status surface from
`nvidia-uccl-ep-runtime-fusion-coordinator-scaffold-status`. It wires private
coordinator state needed to own the PR #176 descriptor allocation and PR #174
runtime path, but it remains narrower than UCCL-EP runtime dispatch and
narrower than pass evidence.

The coordinator scaffold stays private to the CUDA persistent-device runtime
path for one `ChipWorker::run` invocation. It consumes PR #164
same-invocation request args, PR #166 capability metadata, PR #168 validation
policy, PR #170 allocation policy, PR #172 runtime-path map, PR #174
runtime-path scaffold, and PR #176 descriptor allocation scaffold only as
prerequisites. Those prerequisites cannot claim fused success until UCCL-EP
runtime dispatch exists and a fresh H200 fused-boundary result records real
coordinator-owned evidence.

This slice does not implement UCCL-EP runtime dispatch, pass evidence, fresh
H200 fused-success evidence, public API expansion, examples, stable docs,
serving, vLLM, DeepSeek, throughput, or latency.

The accepted surface is private coordinator-owned state for one
`ChipWorker::run` invocation: accepted descriptor allocation, runtime path,
same invocation id, unsupported/failure status, and output sink. It remains
unsupported and provides no runtime dispatch, pass evidence, or H200 fused
success.

## Accepted Runtime Dispatch Scaffold Status Slice

PR #180 merged as `dc32c52dfccfd7838f865a11c3d4837e8ee568ba` and accepted
only the private coordinator-owned runtime-dispatch scaffold/status gate from
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-scaffold-status`. The gate
consumes PR #178 coordinator-owned descriptor allocation and runtime path
state only as prerequisites and records explicit unsupported or failed status
in the runtime-owned output sink.

This branch implements only that private status gate. The gate is owned by
`PtoCudaRuntimeFusionCoordinator` and records whether the coordinator-owned
runtime path is dispatch-scaffold eligible for the same invocation. Missing
gate state is a failed private result with
`missing_runtime_dispatch_scaffold`; eligible gate state remains
`unsupported` and is mirrored into the runtime-owned output sink.

The slice must not run real UCCL-EP dispatch/combine work, emit
scheduler/runtime pass evidence, claim fresh H200 fused success, widen public
`TaskArgs` or `CallConfig`, widen common runtime C API or UCCL host-runtime
ABI fields, add examples or stable docs, or report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

## Runtime Dispatch Request Handoff Map Slice

Branch: `nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-map`.
This runtime dispatch request/driver handoff map is a docs/test dependency
map. It starts after PR #181 at
`05457b7dead2f561be22c24c72771add880f4562`. It defines only the private
UCCL-EP runtime dispatch request/driver handoff from the PR #180
coordinator-owned scaffold/status gate to a later runtime driver. It stays
narrower than scheduler/runtime pass evidence and cannot claim fused success.

Request owner is the private `PtoCudaRuntimeFusionCoordinator`. It may
assemble a future runtime dispatch request only from coordinator-owned
descriptor allocation, private runtime path, validation policy, UCCL-EP
capability metadata, invocation id, PR #180 scaffold/status gate state, and
runtime-owned output sink. Driver owner is a future private UCCL-EP runtime
dispatch driver below the CUDA persistent-device runtime path; driver state
must not come from public TaskArgs, public CallConfig, common runtime C API,
or UCCL host-runtime ABI fields.

Status dependency is the PR #180 runtime-dispatch scaffold/status gate:
missing gate yields `missing_runtime_dispatch_scaffold` and a failed private
result; an eligible prepared gate remains `unsupported`; output is mirrored
to the runtime-owned sink. Failure ownership stays with the coordinator until
a later private driver scaffold accepts the handoff.

Unsupported handoff state covers absent prepared gate, missing request
fields, or missing private driver. Failed handoff state covers stale
invocation id, rank/device mismatch, descriptor-token mismatch, failed
scaffold/status gate, public/API-sourced handoff fields, or fabricated pass
evidence.

This map records no UCCL-EP dispatch/combine work, no scheduler/runtime pass
evidence, no fresh H200 fused success, no public TaskArgs, no public
CallConfig, no common runtime C API, no UCCL host-runtime ABI, no examples,
no stable docs, and no performance claims. It must not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status`.
That future implementation slice may add only private request/driver
handoff scaffold/status plumbing for this map. It remains narrower than pass
evidence and must still avoid real UCCL-EP dispatch/combine work.

## Accepted Runtime Dispatch Request Handoff Scaffold Status Slice

Accepted branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-request-handoff-scaffold-status`.
PR #183 merged as `80b6606282956f38ca6c9a3c52c95d0e5e3a457f` and accepted
only private request/driver handoff scaffold/status plumbing below the
PR #180 coordinator-owned runtime-dispatch gate. The request owner remains
`PtoCudaRuntimeFusionCoordinator`; the private driver placeholder is
coordinator-owned scaffold state, not a real UCCL-EP runtime dispatch driver.

The implemented private ABI surface is
`PtoCudaUcclEpRuntimeDispatchRequestHandoffScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchHandoffDriverState`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_request_handoff_scaffold_status`.
The path validates same invocation id, coordinator-owned runtime path, the
prepared PR #180 gate, request owner, private driver-state pointer, and
runtime-owned output sink. Missing or stale private driver state records
`missing_runtime_dispatch_handoff_driver` and a failed private result. A
valid scaffold/status handoff remains `unsupported` with
`unsupported_boundary`.

This is not UCCL-EP dispatch/combine execution, not scheduler/runtime pass
evidence, and not H200 fused-success evidence. It does not add public
`TaskArgs`, public `CallConfig`, common runtime C API, UCCL host-runtime ABI
fields, examples, stable docs, or performance claims. It must not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` or set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map`. That
docs/test dependency slice may map only private driver-owned
unsupported/failed status vocabulary and failure ownership after this handoff
scaffold. It remains narrower than pass evidence and must not run real
UCCL-EP dispatch/combine work.

## Post-PR183 Status Refresh

Branch:
`nvidia-goal-status-post-runtime-dispatch-handoff-scaffold`.
This status refresh records PR #183 as accepted only for private ABI state
under `PtoCudaRuntimeFusionCoordinator`, same invocation id,
coordinator-owned runtime path/gate, request owner, private driver-state
pointer, runtime-owned output sink, missing/stale handoff driver failure, and
a valid handoff that remains `unsupported`. It changes review-facing
docs/tests only and preserves
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map` as the
single next PR-sized dependency slice.

## Runtime Dispatch Driver Status Map Slice

Branch: `nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-status-map`.
This boundary map defines only the private driver-owned unsupported/failed
status vocabulary and failure ownership after PR #183. PR #183 merged as
`80b6606282956f38ca6c9a3c52c95d0e5e3a457f` and accepted only the
request/driver handoff scaffold/status path.

The private driver owner is the future UCCL-EP runtime dispatch driver below
the CUDA persistent-device runtime path. The failure owner boundary is narrow:
missing driver remains handoff-owned failed while the PR #183 scaffold has not
been accepted by a driver; stale accepted driver is driver-owned failed after
that acceptance. A valid handoff remains `unsupported`, using
`driver_unsupported_boundary`, until real dispatch and combine behavior are
implemented.

Unsupported states:

- `driver_missing`;
- `driver_stale`;
- `driver_not_bound_to_handoff`;
- `driver_no_dispatch_backend`;
- `driver_no_combine_backend`;
- `driver_unsupported_boundary`.

Failed states:

- `driver_owner_mismatch`;
- `driver_invocation_mismatch`;
- `driver_runtime_path_mismatch`;
- `driver_descriptor_token_mismatch`;
- `driver_rank_device_mismatch`;
- `driver_status_sink_mismatch`;
- `driver_public_api_sourced_state`;
- `driver_fabricated_pass_evidence`.

This map records no real UCCL-EP dispatch/combine work, no scheduler/runtime
pass evidence, no fresh H200 fused success, no public `TaskArgs`, no public
`CallConfig`, no common runtime C API, no UCCL host-runtime ABI, and no
examples, stable docs, or performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-scaffold-status`.
That slice may add only a private driver scaffold/status owner for this
vocabulary and remains blocked from real UCCL-EP dispatch/combine work or
pass evidence.

## Runtime Dispatch Driver Scaffold Status Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-scaffold-status`.
This boundary adds only private driver scaffold/status ownership for the
PR #185 vocabulary after PR #185 merged as
`8619767d0eacb5c870b6a56337c6bcb380a2af75`. It is implemented by
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status` in
`src/cuda/platform/include/host/pto_cuda_runtime_fusion_abi.h`.

The scaffold is private to `PtoCudaRuntimeFusionCoordinator`. A valid prepared
driver scaffold must point at the PR #183 handoff status, the
private handoff driver state, the coordinator-owned runtime path, and the
runtime-owned output sink for the same invocation id. That valid state remains
`unsupported` with `driver_unsupported_boundary`; it does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Malformed or mismatched private driver scaffold/status is failed private
state. The driver-owned failure names are `driver_owner_mismatch`,
`driver_invocation_mismatch`, `driver_runtime_path_mismatch`,
`driver_descriptor_token_mismatch`, `driver_rank_device_mismatch`,
`driver_status_sink_mismatch`, `driver_public_api_sourced_state`, and
`driver_fabricated_pass_evidence`.

No public `TaskArgs`, public `CallConfig`, common runtime C API, UCCL
host-runtime ABI, examples, stable docs, performance claims, real UCCL-EP
dispatch/combine work, scheduler/runtime pass evidence, or fresh H200 fused
success are added by this slice.

Selected next slice:
`nvidia-goal-status-post-runtime-dispatch-driver-scaffold-status`. It is a
review-facing status refresh only.

## Post-Runtime-Dispatch-Driver-Scaffold Status Refresh

Branch:
`nvidia-goal-status-post-runtime-dispatch-driver-scaffold-status`.
PR #186 merged as `7589e2df44ad4df9c200cd4ec673dacac0a27a71`
(`Add runtime dispatch driver scaffold status`) and is accepted only for
private runtime-dispatch driver scaffold/status ownership.

The accepted private runtime boundary is
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`, driver-owned failure bits,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`,
and the host runtime private call. The valid status remains unsupported with
`driver_unsupported_boundary`; malformed/mismatched produces failed private
result with driver-owned failures.

This status refresh changes review-facing docs/tests only. It records no real
UCCL-EP dispatch/combine work, no scheduler/runtime pass evidence, no fresh
H200 fused success, no public `TaskArgs`, no public `CallConfig`, no common
runtime C API, no UCCL host-runtime ABI, and no examples, stable docs, or
performance claims. It does not report
`persistent_device_uccl_ep_runtime_fusion.status: passed` and does not set
`actual_fused_cross_gpu_execution: true`.

Selected next slice:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-map`.
This is selected exactly one next PR-sized dependency map slice for real
runtime dispatch driver request/backend ownership. It is not
implementation/pass evidence. The next slice must stay narrower than real
UCCL-EP dispatch/combine work, scheduler/runtime pass evidence, and H200
fused-success claims.

## Runtime Dispatch Driver Backend Map Slice

Branch:
`nvidia-uccl-ep-runtime-fusion-runtime-dispatch-driver-backend-map`.

This dependency map records the future private runtime dispatch driver's
request/backend ownership boundary after PR #186
(`7589e2df44ad4df9c200cd4ec673dacac0a27a71`). PR #186 accepted only
`PtoCudaUcclEpRuntimeDispatchDriverScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverStatus`, driver-owned failure bits, and
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_scaffold_status`.
The valid prepared driver scaffold remains `unsupported`; it is not backend
execution or pass evidence.

Boundary ownership:

- private driver request owner: the future private driver accepts the
  scaffold only after the coordinator-owned handoff, invocation id, runtime
  path, descriptor token, rank/device map, and runtime-owned output sink
  match;
- dispatch backend placeholder: a driver-owned placeholder for future
  UCCL-EP dispatch, with no transport calls, payload transfer, kernel launch,
  scheduler transition, or pass evidence;
- combine backend placeholder: a driver-owned placeholder for future UCCL-EP
  combine, with no reduce/combine transport, payload release, kernel launch,
  scheduler transition, or pass evidence;
- status sink owner: the runtime-owned output sink remains the only
  review-facing sink; the driver can record driver status and failure names
  there but cannot source state from example JSON, adapter-only provenance,
  public `TaskArgs`, public `CallConfig`, common runtime C API, or UCCL
  host-runtime ABI fields;
- driver-owned failure propagation: after the driver accepts the valid
  scaffold, backend request/backend/status-sink mismatches are driver-owned
  failed states.

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

The invalid pass-evidence boundary rejects example JSON, adapter-only
provenance, public `TaskArgs`, public `CallConfig`, common runtime C API
fields, UCCL host-runtime ABI fields, and hand-authored review artifacts as
backend pass sources. This slice records no real UCCL-EP dispatch/combine
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
(`7bc598f75d5738193a7b53fa10a751f2518edb17`). It keeps backend scaffold
status behind `persistent_device_uccl_ep_runtime_fusion_entry` and does not
add public `TaskArgs`, public `CallConfig`, common runtime C API, UCCL
host-runtime ABI, examples, stable docs, or performance claims.

Implementation evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_scaffold_status_is_driver_owned`.

The valid prepared backend scaffold/status remains `unsupported`,
`actual_fused_cross_gpu_execution` remains `0`, and no passed status,
transport call, dispatch backend execution, or combine backend execution is
reported. The private scaffold records backend owner, invocation id, runtime
path, runtime-owned status sink, descriptor token, and rank/device/world
metadata only.

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
The accepted backend scaffold/status remains `unsupported`; it is not a
backend request, dispatch request, combine request, or pass-evidence source.

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
(`4223edd9fa3c5e58b62eff1d7c27b1a54670766d`). It keeps the backend request
scaffold/status behind `persistent_device_uccl_ep_runtime_fusion_entry` and
does not add public `TaskArgs`, public `CallConfig`, common runtime C API,
UCCL host-runtime ABI, examples, stable docs, or performance claims.

Implementation evidence:

- `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus`;
- `PtoCudaUcclEpRuntimeDispatchDriverBackendRequestStatus`;
- `pto_cuda_uccl_ep_runtime_dispatch_driver_backend_request_status_name`;
- `pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_request_scaffold_status`;
- `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_REQUEST_SCAFFOLD`;
- `test_private_runtime_dispatch_driver_backend_request_scaffold_status_is_backend_owned`.

The valid prepared backend request scaffold/status remains `unsupported`,
`actual_fused_cross_gpu_execution` remains `0`, and no passed status,
transport call, dispatch request execution, or combine request execution is
reported. The private scaffold records backend request owner, backend
scaffold/status input, invocation id, runtime path, runtime-owned status
sink, descriptor token, and rank/device/world metadata only.

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
(`14aaedd8865ea7351cd30ee1a0dc46804b7d0f36`). It adds the dispatch request
scaffold/status vocabulary behind the CUDA runtime boundary:
`PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestScaffoldStatus`,
`PtoCudaUcclEpRuntimeDispatchDriverBackendDispatchRequestStatus`,
`pto_cuda_uccl_ep_runtime_dispatch_driver_backend_dispatch_request_status_name`,
`pto_cuda_runtime_fusion_prepare_runtime_dispatch_driver_backend_dispatch_request_scaffold_status`,
and `PTO_CUDA_RUNTIME_FUSION_FAILURE_DRIVER_BACKEND_DISPATCH_REQUEST_SCAFFOLD`.

The scaffold is coordinator-owned private state adjacent to
`PtoCudaUcclEpRuntimeDispatchDriverBackendRequestScaffoldStatus`. Its prepared
status remains `unsupported`, preserves `actual_fused_cross_gpu_execution` as
`0`, and reports failures only through the private runtime-fusion result sink.
It does not expose dispatch request state through public `TaskArgs`, public
`CallConfig`, common runtime C API fields, UCCL host-runtime ABI fields,
examples, stable docs, or hand-authored review artifacts.

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

Implementation evidence is
`test_private_runtime_dispatch_driver_backend_dispatch_request_scaffold_status_is_backend_owned`.
Malformed dispatch request scaffold/status state fails privately; a valid
prepared dispatch request scaffold/status remains `unsupported`. This is no real UCCL-EP
dispatch/combine work, no scheduler/runtime pass evidence, no fresh H200
fused success, no examples, stable docs, or performance claims. It does not
report `persistent_device_uccl_ep_runtime_fusion.status: passed` and does not
set `actual_fused_cross_gpu_execution: true`.
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
(`562778f051ca87cf3f62d796860a8fd4c3476a32`) without crossing the
communication runtime boundary into real UCCL-EP dispatch/combine work.

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

This boundary note records private combine payload descriptor scaffold/status
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

## Non-Claims

This slice does not claim UCCL host-runtime dispatch, RDMA, multi-node
transport, serving integration, vLLM integration, or DeepSeek model
correctness. The UCCL-EP handoff is adapter/probe evidence only. Later slices
must add fresh implementation, evidence, and docs before making broader
communication or serving claims.
DeepSeek model correctness remains out of scope.
