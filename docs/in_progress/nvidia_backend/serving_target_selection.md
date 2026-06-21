# Serving Target Selection

This note selects the first serving integration target for the NVIDIA backend
restart. It is based on local source reads from the manifest, not serving
evidence, and not DeepSeek correctness.

## Sources Read

| Source | Local Path | Relevant Finding |
| ------ | ---------- | ---------------- |
| pypto-serving | `tmp/sources/repos/hw-native-sys/pypto-serving` | Small PTO-owned serving stack with an OpenAI-compatible API, `ModelExecutor`, `PyptoExecutor`, and `python/runtime/worker.py` wrapping Simpler workers. Current implementation is Ascend/Qwen3-oriented. |
| vLLM | `tmp/sources/repos/external/vllm` | Mature NVIDIA/Hugging Face serving stack with OpenAI-compatible API, plugin loading through `vllm.plugins`, platform/custom-op surfaces, MoE support, and `vllm._custom_ops`. |
| FlashInfer | `tmp/sources/repos/external/flashinfer` | FlashInfer kernel reference for serving attention, GEMM, MoE, FP8/FP4 paths, H200 support, and distributed communication features. |
| DeepSeek V4 deployment article | `tmp/sources/web/deepseek-v4-self-hosting-guide-vllm-hardware-deployment.html` | DeepSeek-V4-Flash compatibility baseline source that frames vLLM deployment, 2x H200 weight-fit expectations, expert parallelism, and quantization context. |

## Decision

Use `pypto-serving` as the first PTO-owned integration target for launching
simpler NVIDIA kernels from a serving loop.

Use vLLM as the DeepSeek-V4-Flash compatibility baseline and later external
integration target, not as the first place to wire unstable simpler-nv kernels.
Use FlashInfer as the serving-kernel reference, not as a serving framework.

## Rationale

`pypto-serving` already has the internal shape PTO needs for an early
simpler-nv integration:

- `python/core/executor.py` defines a backend-neutral `ModelExecutor` with
  `run_prefill()` and `run_decode()` boundaries.
- `python/core/pypto_executor.py` defines `PyptoExecutor`, which compiles once
  and delegates runtime execution to model-specific runners.
- `python/runtime/worker.py` wraps `simpler.worker.Worker` and exposes worker
  lifecycle, callable dispatch, child-memory tensors, and tensor handles.
- The HTTP layer is already OpenAI-compatible, so a local smoke can exercise a
  serving request without adopting all of vLLM first.

vLLM should still remain central to the goal:

- It is the mature NVIDIA serving path for DeepSeek-V4-Flash compatibility.
- It already has plugin loading through `vllm.plugins`, custom-op access
  through `vllm._custom_ops`, NVIDIA kernel imports through the current
  platform, and built-in MoE/attention infrastructure.
- It is the right external baseline for final DeepSeek-V4-Flash serving on 2
  H200 GPUs.

FlashInfer should guide the kernel surface:

- Its README names attention, GEMM, MoE, FP8/FP4, H200, and serving decode /
  prefill kernels as core scope.
- PTO's Gluon and persistent-kernel work should compare future serving kernels
  against FlashInfer-style operator coverage before claiming serving readiness.

## First Integration Slice

The first serving child slice should add a local simpler-nv executor shim for
`pypto-serving`, not a full DeepSeek model:

1. Reuse the `ModelExecutor` / `PyptoExecutor` boundary shape.
2. Replace the Ascend-specific runner with a minimal CUDA simpler runner that
   can call an already-verified generated kernel or persistent MoE seed.
3. Keep model loading small: synthetic tensors or a tiny fixture are enough for
   the first HTTP request path.
4. Add a local skip-safe test for CLI/config wiring and a remote H200 command
   only after the local serving loop can launch the simpler CUDA kernel.

## Non-Claims

- This is not serving evidence.
- This is not DeepSeek correctness.
- This is not a vLLM plugin implementation.
- This is not a claim that `pypto-serving` can run DeepSeek-V4-Flash today.
- This is not a claim that FlashInfer kernels have been integrated into PTO.

## Follow-Up Gates

- Add a `pypto-serving` shim design before editing serving code.
- Prove a single OpenAI-compatible local request can launch a simpler CUDA
  kernel.
- Compare the request path against vLLM's DeepSeek-compatible behavior before
  attempting DeepSeek-V4-Flash.
- Use FlashInfer as the operator checklist for attention, KV cache, sampling,
  GEMM, MoE, FP8/FP4, and decode/prefill coverage.
