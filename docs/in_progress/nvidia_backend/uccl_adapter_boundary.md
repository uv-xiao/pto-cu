# UCCL Adapter Boundary

This note defines the PR3 UCCL adapter boundary. It records
adapter/probe evidence only; it is not a CUDA host-runtime communication
contract.

## Decision

Keep UCCL behind the existing private CUDA communication helper:
`simpler_setup/cuda_comm.py`.

- `CudaCommCapability(backend="uccl")` names an opaque rank/device group.
- `UcclP2PWriteIpcDescriptor` records `src_rank`, `dst_rank`, and `nbytes`
  for same-node IPC writes.
- `UcclEpDispatchCombineDescriptor` records MoE dispatch/combine metadata:
  token count, hidden size, top-k count, expert count, dtype, and metadata
  tensor shapes.
- `UcclP2PCudaCommRuntime` wraps a local `uccl.p2p.Endpoint` for Python-side
  adapter probes.
- `CudaCommRuntimeRegistry` may acquire that Python-side P2P adapter with
  `uccl_transport="p2p_ipc"`.

No CUDA host-runtime UCCL ABI is added in this PR. `TaskArgs` and
`CallConfig` remain unchanged, and no public Python UCCL handles are exposed.

## Surface Split

UCCL-P2P and UCCL-EP are separate review surfaces:

| Surface | PTO role | Current evidence |
| ------- | -------- | ---------------- |
| UCCL-P2P | same-node explicit data movement probe | Python-side IPC adapter |
| UCCL-EP | MoE dispatch/combine metadata probe | Python-side descriptor and skip-safe harness |
| UCCL collectives | possible later comparison path | not implemented here |

The current CUDA host runtime continues to own NCCL operation dispatch. UCCL
stays at the Python descriptor and probe layer until a later PR selects a
stable C or C++ runtime boundary.

## Non-Claims

RDMA is not proven. multi-node is not proven. serving integration is not proven.
DeepSeek correctness is not proven. CUDA host-runtime UCCL operation dispatch
is not proven.

The restart/control checkout contains historical restart context for UCCL
package and adapter runs. Those runs are useful context, but they are not fresh
PR evidence for this branch unless rerun and recorded by this PR.

Fresh PR evidence now covers same-node H200 UCCL-P2P IPC adapter execution and
reduced-shape H200 UCCL-EP dispatch/combine adapter execution through the
Python-side examples. The direct EP command without `UCCL_EP_BENCH_DIR`
correctly returned structured skip JSON with `--require-cuda`; the
dependency-qualified command passed after pointing at the external UCCL bench
helper.
