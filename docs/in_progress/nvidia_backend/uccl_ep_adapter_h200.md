# UCCL EP Adapter H200 Evidence

This note tracks PTO's private UCCL-EP dispatch/combine descriptor probe. It
is adapter/probe evidence only: not RDMA evidence, not multi-node evidence,
not serving evidence, and not DeepSeek correctness evidence.

Boundary phrases: not RDMA evidence; not multi-node evidence; not serving evidence.

## Harness

- Example: `examples/cuda/uccl_ep_dispatch_combine_adapter.py`
- Private PTO operation name: `ep_dispatch_combine`
- Descriptor: `UcclEpDispatchCombineDescriptor`
- Capability: `CudaCommCapability(backend="uccl")`
- Optional external helper: UCCL EP bench `Buffer` from `UCCL_EP_BENCH_DIR`

The descriptor records MoE routing metadata and shape constraints. The harness
maps that descriptor into installed `uccl.ep` only when optional dependencies,
CUDA, `torch.distributed`, and the external bench helper are available.

## Historical Restart Context

The restart/control checkout recorded historical restart context for reduced
H200 BF16 and FP8 UCCL-EP adapter smokes. Those results motivate this PR but
are not fresh PR evidence for this branch.

## Fresh PR Evidence

Fresh PR evidence came from a synced remote tree of branch
`cuda-uccl-adapter-evidence` on 2026-06-21.

The required direct command returned structured skip JSON because the synced
checkout does not include the external UCCL EP bench helper:

```text
status: skipped
reason: UCCL-EP bench buffer.py unavailable; set
  UCCL_EP_BENCH_DIR=/path/to/uccl/ep/bench
exit status: 2
```

After pointing `UCCL_EP_BENCH_DIR` at the existing remote UCCL bench checkout,
the adapter passed on H200 devices `6,7`:

```text
status: passed
transport: ep
operation: ep_dispatch_combine
input_dtype: bf16
world_size: 2
num_tokens: 64
hidden: 128
num_topk: 4
num_experts: 16
rank 0 recv_tokens: [88]
rank 0 max_abs_error: 0.0
rank 0 topk_weight_error: 0.0
rank 1 recv_tokens: [88]
rank 1 max_abs_error: 0.0
rank 1 topk_weight_error: 0.0
```

This is reduced-shape intranode UCCL-EP adapter evidence. It is not a
performance claim and does not prove serving integration.

## Persistent MoE Handoff Evidence

Fresh PR evidence for `persistent-moe-uccl-ep-handoff` composes this adapter
with the existing two-device persistent MoE aggregate. The handoff command ran
on the same H200 devices `6,7` with `hidden: 1024`, `num_tokens: 64`,
`num_topk: 4`, `num_experts: 16`, and `input_dtype: bf16`.

Result: pass.

- `handoff_scope`: `persistent-moe-plus-uccl-ep-adapter`
- `persistent_moe.status`: `passed`
- `uccl_ep_adapter.status`: `passed`
- `uccl_ep_adapter.transport`: `ep`
- `uccl_ep_adapter.operation`: `ep_dispatch_combine`
- `uccl_ep_adapter.descriptor.operation`: `ep_dispatch_combine`
- `uccl_ep_adapter.descriptor.hidden`: `1024`
- `uccl_ep_adapter.descriptor.metadata_shapes.topk_idx`: `[64, 4]`
- `uccl_ep_adapter_max_abs_error`: `0.0`
- `uccl_ep_adapter_topk_weight_error`: `0.0`
- `handoff_validation.same_device_ids`: `true`
- `handoff_validation.adapter_descriptor_metadata_present`: `true`
- `handoff_validation.max_errors_zero`: `true`

The embedded adapter payload sanitizes local bench and Python-package paths.
This remains Python-side adapter/probe evidence. It does not prove CUDA
host-runtime UCCL dispatch, RDMA, multi-node transport, serving integration,
DeepSeek correctness, or performance.

## Non-Claims

No CUDA host-runtime UCCL ABI is added. No serving integration is proven.
No multi-node or RDMA behavior is proven.
