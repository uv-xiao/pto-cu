# UCCL P2P Adapter H200 Evidence

This note tracks the PTO-side UCCL-P2P adapter probe. It is not RDMA evidence,
not multi-node evidence, not serving evidence, and not DeepSeek correctness
evidence.

Boundary phrases: not RDMA evidence; not multi-node evidence; not serving evidence.

## Adapter Under Test

- Example: `examples/cuda/uccl_p2p_ipc_adapter.py`
- Private runtime wrapper: `UcclP2PCudaCommRuntime`
- Operation descriptor: `UcclP2PWriteIpcDescriptor`
- Transport selector: `uccl_transport="p2p_ipc"`
- Descriptor fields: `src_rank`, `dst_rank`, and `nbytes`

The example exchanges endpoint metadata, connects local ranks through
`uccl.p2p`, advertises destination IPC memory, and writes from the source rank.

## Historical Restart Context

The restart/control checkout recorded historical restart context for a
same-node H200 P2P adapter run. That context is useful for adapter design, but
it is not fresh PR evidence for this branch.

## Fresh PR Evidence

Fresh PR evidence came from a synced remote tree of branch
`cuda-uccl-adapter-evidence` on 2026-06-21.

Command:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  bash -lc 'source .venv/bin/activate && \
    CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 \
    PYTHONPATH=$PWD:$PWD/python \
    torchrun --standalone --nproc_per_node=2 \
      examples/cuda/uccl_p2p_ipc_adapter.py \
      --device-ids 0,1 --nbytes 1024 --require-cuda'
```

Result summary:

```text
status: passed
backend: uccl
transport: p2p_ipc
operation: p2p_write_ipc
world_size: 2
CUDA_VISIBLE_DEVICES: 6,7
device_ids: [0, 1]
nbytes: 1024
rank 0 role: client, passed: true
rank 1 role: server, passed: true
```

The UCCL log reported RDMA device discovery, but this adapter command used the
same-node `write_ipc` path. Treat this result as same-node IPC evidence only.

## Non-Claims

No CUDA host-runtime UCCL ABI is added. This note does not prove RDMA,
multi-node behavior, serving integration, or DeepSeek correctness.
