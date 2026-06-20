# NCCL Two-H200 Baseline

This note records the first NCCL baseline for the NVIDIA backend restart. It is
evidence for the 2 H200 communication milestone, not a PTO runtime integration
claim.

## Harness

- Script: `examples/cuda/nccl_two_gpu_baseline.py`
- Backend: NCCL through `simpler_setup.cuda_comm.CudaCommRuntimeRegistry`
  and `torch.distributed`
- World size: `world_size: 2`
- Devices: `0,1`
- Capability id: `nccl:rank0->cuda0,rank1->cuda1`
- Launch plan runtime ids:
  `nccl:rank0->cuda0,rank1->cuda1/local_rank0` and
  `nccl:rank0->cuda0,rank1->cuda1/local_rank1`
- Tensor size: `1024` FP32 elements
- Operations: `all_reduce`, `reduce_scatter`, `all_gather`, `send_recv`

The script is skip-safe on machines without PyTorch, CUDA, two visible GPUs, or
NCCL support. The H200 run used `--require-cuda`, so a skip would have returned
a non-zero status.

## H200 Environment

Command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total \
  --format=csv,noheader
```

Distilled output:

```text
0, NVIDIA H200 NVL, 580.126.20, 139641 MiB, 143771 MiB
1, NVIDIA H200 NVL, 580.126.20, 1383 MiB, 143771 MiB
2, NVIDIA H200 NVL, 580.126.20, 79552 MiB, 143771 MiB
3, NVIDIA H200 NVL, 580.126.20, 79538 MiB, 143771 MiB
4, NVIDIA H200 NVL, 580.126.20, 79537 MiB, 143771 MiB
5, NVIDIA H200 NVL, 580.126.20, 79547 MiB, 143771 MiB
6, NVIDIA H200 NVL, 580.126.20, 79537 MiB, 143771 MiB
7, NVIDIA H200 NVL, 580.126.20, 26487 MiB, 143771 MiB
```

PyTorch capability check:

```json
{
  "cuda_available": true,
  "device_count": 8,
  "distributed_available": true,
  "nccl_available": true,
  "torch": "2.8.0+cu128"
}
```

Topology snapshot:

```text
GPU0 to GPU1: NV6
GPU0 to GPU2: NV6
GPU0 to GPU3: NV6
GPU0 to GPU4: SYS
GPU4 to GPU5: NV6
```

The first run used devices `0,1`, which are both `NVIDIA H200 NVL` and connected
by `NV6`.

## Result

Command:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  env NCCL_DEBUG=WARN \
  .venv/bin/python examples/cuda/nccl_two_gpu_baseline.py \
  --device-ids 0,1 --tensor-numel 1024 --require-cuda
```

Distilled output:

```text
NCCL version 2.27.3+cuda12.9
status: passed
world_size: 2
device_ids: [0, 1]
tensor_numel: 1024
elapsed_s: 13.86473035812378
capability_id: nccl:rank0->cuda0,rank1->cuda1
rank_to_device: {"0": 0, "1": 1}
launch_plans:
  - rank 0, device_id 0, runtime_id nccl:rank0->cuda0,rank1->cuda1/local_rank0
  - rank 1, device_id 1, runtime_id nccl:rank0->cuda0,rank1->cuda1/local_rank1
operations: [all_reduce, reduce_scatter, all_gather, send_recv]
all_reduce: passed, checksum 1048576.0 on both ranks
reduce_scatter: passed, checksum 1048576.0 on rank 0 and 1069056.0 on rank 1
all_gather: passed, values [0.0, 1.0] on both ranks
send_recv: passed, rank 0 received 23.0 and rank 1 received 17.0
```

## Interpretation

This establishes a working NCCL compatibility floor for 2 H200 communication:
basic collective, reduce-scatter/all-gather, and point-to-point paths work on
the selected local H200 pair, and the result now carries the same opaque
capability metadata as `simpler_setup/cuda_comm.py`. The script now exercises
the internal `CudaCommHostPlan` and `CudaCommRuntimeRegistry` NCCL path,
including local-rank launch plans, process-group setup, and teardown. It does
not yet prove a PTO
host-runtime integration, UCCL expert-parallel behavior, fused MoE
dispatch/combine, multi-node transport, or serving-level DeepSeek-V4-Flash
behavior.

Next communication work should keep this NCCL script as the baseline while
probing the PTO runtime boundary and UCCL EP/P2P behavior on the same device
pair.
