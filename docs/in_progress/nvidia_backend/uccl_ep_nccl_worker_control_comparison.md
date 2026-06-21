# UCCL EP And NCCL Worker-Control Comparison

This note compares evidence roles. It is not RDMA evidence, not multi-node
evidence, not serving evidence, and not a performance claim.

Boundary phrases: not RDMA evidence; not multi-node evidence; not serving evidence.

## Historical Restart Context

The restart/control checkout recorded historical restart context comparing a
UCCL-EP adapter shape with a larger NCCL worker-control payload run. That
context is not fresh PR evidence for this branch.

## Fresh PR Evidence

Fresh PR evidence for PR3 is local descriptor and skip-safe adapter coverage
plus H200 UCCL-P2P and UCCL-EP adapter probes from branch
`cuda-uccl-adapter-evidence`. The established fresh baseline from PR #59
remains NCCL worker-control operation dispatch through
`examples/cuda/nccl_worker_control_ops.py`.

The fresh UCCL-EP run used `CUDA_VISIBLE_DEVICES=6,7`, `world_size=2`,
`num_tokens=64`, `hidden=128`, `num_topk=4`, and `num_experts=16`. Both ranks
reported `max_abs_error: 0.0` and `topk_weight_error: 0.0`.

## Role Comparison

| Harness | Role |
| ------- | ---- |
| `examples/cuda/uccl_ep_dispatch_combine_adapter.py` | Python-side UCCL-EP descriptor/probe harness |
| `examples/cuda/nccl_worker_control_ops.py` | CUDA host-runtime NCCL worker-control baseline |

UCCL-EP `ep_dispatch_combine` is not semantically equivalent to NCCL
`all_reduce`, `reduce_scatter`, `all_gather`, or `send_recv`. The comparison
only keeps the current NCCL baseline visible while UCCL remains Python-side
adapter/probe evidence.

## Non-Claims

No CUDA host-runtime UCCL ABI is added. This comparison does not prove RDMA,
multi-node behavior, serving integration, or DeepSeek correctness.
