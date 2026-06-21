# UCCL EP/P2P H200 Probe Evidence

This note tracks H200 UCCL package probe context for PR3. It is not RDMA
evidence, not multi-node evidence, not serving evidence, and not DeepSeek
correctness evidence.

Boundary phrases: not RDMA evidence; not multi-node evidence; not serving evidence.

## Historical Restart Context

The restart/control checkout recorded historical restart context where
external UCCL sources were built on a remote H200 environment and reduced
same-node probes were attempted. That context indicated:

- `uccl.p2p` could be built/imported after local build environment fixes.
- `uccl.ep` could be built/imported with DMA-BUF enabled.
- Same-node IPC and reduced intranode EP dispatch/combine smokes were reported
  in that workspace.

Those results are not fresh PR evidence for this branch. They are included
only to explain why PR3 preserves Python-side descriptors and skip-safe
adapter harnesses.

## Fresh PR Evidence

Fresh PR evidence came from a `--sync` remote tree copy of branch
`cuda-uccl-adapter-evidence` into the H200 checkout on 2026-06-21.

Remote snapshot:

```text
GPUs: NVIDIA H200 NVL, devices 0-7
compute capability: 9.0
driver: 580.126.20
python: 3.12.3
torch: 2.8.0+cu128
torch CUDA: 12.8
uccl.p2p import: ok
uccl.ep import: ok
```

The required command without an explicit `UCCL_EP_BENCH_DIR` returned status
`2` because the synced checkout intentionally does not include gitignored
external sources:

```text
status: skipped
reason: UCCL-EP bench buffer.py unavailable; set
  UCCL_EP_BENCH_DIR=/path/to/uccl/ep/bench
```

The dependency-qualified UCCL-EP command used the existing remote UCCL bench
checkout and passed:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh -- \
  bash -lc 'source .venv/bin/activate && \
    UCCL_EP_BENCH_DIR=/tmp/uccl-codex-probe/ep/bench \
    CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 \
    PYTHONPATH=$PWD:$PWD/python \
    torchrun --standalone --nproc_per_node=2 \
      examples/cuda/uccl_ep_dispatch_combine_adapter.py \
      --device-ids 0,1 --num-tokens 64 --hidden 128 \
      --num-topk 4 --num-experts 16 --input-dtype bf16 \
      --require-cuda'
```

Result summary:

```text
status: passed
transport: ep
operation: ep_dispatch_combine
input_dtype: bf16
world_size: 2
CUDA_VISIBLE_DEVICES: 6,7
rank 0 recv_tokens: [88]
rank 0 max_abs_error: 0.0
rank 0 topk_weight_error: 0.0
rank 1 recv_tokens: [88]
rank 1 max_abs_error: 0.0
rank 1 topk_weight_error: 0.0
```

The UCCL-P2P adapter also passed on the same H200 pair; see
`uccl_p2p_adapter_h200.md`.

## Non-Claims

This note does not prove RDMA, multi-node behavior, serving integration,
CUDA host-runtime UCCL operation dispatch, or DeepSeek correctness.
