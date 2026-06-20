# Distributed Communication Selection

This note records the first Ray/NCCL/UCCL selection for the NVIDIA backend
restart. It is a planning boundary, not performance evidence.

## Role Summary

| Source | Role For PTO NVIDIA | Initial Decision |
| ------ | ------------------- | ---------------- |
| Ray | Process and serving orchestration | Keep as an orchestration option outside the compiled CUDA scheduler and worker path. |
| NCCL | Baseline collectives | Use as the first baseline for all-reduce, all-gather, reduce-scatter, broadcast, and send/receive comparisons on NVIDIA GPUs. |
| UCCL | EP/P2P research path | Keep as a later experimental direction. Do not include UCCL host-runtime dispatch or adapter execution in this NCCL worker-control slice. |

## Implications

- The simpler NVIDIA platform should expose communication through a narrow
  runtime boundary. The persistent-device megakernel may orchestrate device
  work, but it should not directly depend on Ray.
- NCCL is the default measurement baseline because it is the standard NVIDIA
  GPU collective library and covers both single-node and multi-node
  collectives.
- UCCL remains a later research path for expert-parallel and P2P work. It is
  not part of this PR's implementation or execution evidence.
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
4. Compare UCCL P2P or expert-parallel paths in later slices after the NCCL
   worker-control baseline is accepted.
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

## Non-Claims

UCCL PTO host-runtime dispatch, UCCL adapter execution, RDMA evidence,
multi-node evidence, serving-level communication evidence, and DeepSeek model
correctness remain pending.
