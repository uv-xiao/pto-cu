# UCCL EP/P2P Probe Plan

This plan describes the UCCL adapter/probe evidence expected from PR3. It is
descriptor/adapter-preparation for the NVIDIA backend, not a host-runtime ABI
plan.

## Scope

- Preserve UCCL-P2P and UCCL-EP experiments as Python-side evidence.
- Keep UCCL source and build products outside git.
- Keep `TaskArgs`, `CallConfig`, and CUDA host-runtime public ABI unchanged.
- Keep NCCL worker-control behavior as the current host-runtime baseline.

## Probe Order

1. Run local unit tests for `simpler_setup.cuda_comm` descriptor validation,
   fake UCCL runtime behavior, and skip-safe example CLI behavior.
2. Run `examples/cuda/uccl_p2p_ipc_adapter.py` only when UCCL-P2P, CUDA, and
   `torchrun` are available.
3. Run `examples/cuda/uccl_ep_dispatch_combine_adapter.py --require-cuda`
   when UCCL-EP dependencies are available.
4. If the H200 UCCL packages are unavailable, record the exact failure and
   keep this PR honest as descriptor/adapter-preparation.

Required remote command for the EP gate:

```bash
REMOTE_PTO_CU=/tmp/pto-cu-codex-restart \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python .venv/bin/python \
      examples/cuda/uccl_ep_dispatch_combine_adapter.py --require-cuda'
```

## Evidence Rules

Each UCCL note must say whether evidence is fresh PR evidence or historical
restart context. RDMA, multi-node, serving, DeepSeek, and CUDA host-runtime
UCCL ABI claims are out of scope.

No CUDA host-runtime UCCL ABI is introduced by this plan; in prompt terms,
this is no CUDA host-runtime UCCL ABI.
