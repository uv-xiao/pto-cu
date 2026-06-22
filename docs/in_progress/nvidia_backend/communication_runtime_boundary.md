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

## Non-Claims

This slice does not claim UCCL host-runtime dispatch, RDMA, multi-node
transport, serving integration, vLLM integration, or DeepSeek model
correctness. The UCCL-EP handoff is adapter/probe evidence only. Later slices
must add fresh implementation, evidence, and docs before making broader
communication or serving claims.
DeepSeek model correctness remains out of scope.
