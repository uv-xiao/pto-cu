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
- The next implementation PR should keep the current H200 command shape and
  convert only the fused-boundary result from `unsupported` to `passed` after
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
  remains a private entry-contract only. None of those PRs proves actual
  fused cross-GPU expert-parallel MoE execution.
- The next PR-sized slice is
  `nvidia-uccl-ep-runtime-fusion-private-entry-unsupported`. It can implement
  only private entry scaffolding behind `ChipWorker::run` /
  `ChipStorageTaskArgs` and must keep the fused-boundary result `unsupported`
  unless the runtime coordinator emits real descriptor ownership,
  ownership-token, lifetime-transition, rank/device, validation, and
  failure-field evidence.

## Non-Claims

UCCL PTO host-runtime dispatch, RDMA evidence, multi-node evidence,
serving-level communication evidence, and DeepSeek model correctness remain
pending. The UCCL-EP handoff evidence is adapter/probe evidence only.
UCCL adapter execution is limited to opt-in Python-side probes and handoff
gates. PR #153 did not change CUDA runtime behavior, claim fresh H200 fused
success, report
`persistent_device_uccl_ep_runtime_fusion.status: passed`, or set
`actual_fused_cross_gpu_execution: true`.
