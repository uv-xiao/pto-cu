# pypto-serving simpler-nv Shim Design

This design follows `serving_target_selection.md`: use `pypto-serving` as the
first PTO-owned serving integration target, while keeping vLLM as the
DeepSeek-V4-Flash compatibility baseline. It is a design note only: no
DeepSeek-V4-Flash claim, no vLLM plugin claim, and no serving evidence.

## Source Contracts

The shim targets these `pypto-serving` contracts:

| Contract | Source | Required Shape |
| -------- | ------ | -------------- |
| `ModelExecutor` | `tmp/sources/repos/hw-native-sys/pypto-serving/python/core/executor.py` | Implements `run_prefill()` and `run_decode()` over `RuntimeModel`, `PrefillBatch`, and `DecodeBatch`. |
| `PyptoExecutor` | `tmp/sources/repos/hw-native-sys/pypto-serving/python/core/pypto_executor.py` | Compiles/registers model artifacts once, creates a model-specific `ModelRunner`, and delegates prefill/decode. |
| `ModelRunner` | `tmp/sources/repos/hw-native-sys/pypto-serving/python/core/model_runner.py` | Owns worker-resident KV cache allocation plus `run_prefill()` and `run_decode()`. |
| Engine records | `tmp/sources/repos/hw-native-sys/pypto-serving/python/core/types.py` | Carries `RuntimeConfig`, `RuntimeModel`, `PrefillBatch`, `DecodeBatch`, and result dataclasses. |
| Serving loop | `tmp/sources/repos/hw-native-sys/pypto-serving/python/core/engine.py` | Builds prompt embeddings, allocates KV pages, calls executor prefill/decode, and samples logits. |
| HTTP API | `tmp/sources/repos/hw-native-sys/pypto-serving/python/core/server.py` | Provides OpenAI-compatible API routes without needing NVIDIA-specific changes for the first shim. |
| Worker wrapper | `tmp/sources/repos/hw-native-sys/pypto-serving/python/runtime/worker.py` | Wraps `simpler.worker.Worker` and exposes worker lifecycle, callable dispatch, and child-memory tensor handles. |

## Minimal CUDA Runner

The first implementation should add a minimal CUDA runner parallel to the
existing Ascend/Qwen runner, not a full model port:

1. Define a `SimplerNvExecutor` following the `ModelExecutor` shape.
2. Define a `SimplerNvModelRunner` following the `ModelRunner` shape.
3. Use `python/runtime/worker.py`'s wrapper style but instantiate
   `simpler.worker.Worker(platform="cuda", runtime=...)`.
4. Register one already-verified CUDA callable or generated kernel from the
   current NVIDIA restart seeds.
5. Return deterministic logits from `run_prefill()` and `run_decode()` so the
   existing `LLMEngine` and OpenAI-compatible API can complete a synthetic
   request.

The first shim should use a synthetic model fixture. It should not load
DeepSeek weights, and it should not require vLLM.

## Runtime Boundary

The shim must keep the existing serving boundaries intact:

- `LLMEngine` still owns tokenization, KV allocation, request state, and
  sampling.
- `ModelExecutor` still owns backend-specific validation and prefill/decode.
- `ModelRunner` still owns worker-resident KV cache and callable dispatch.
- `python/runtime/worker.py` remains the only layer that talks directly to
  `simpler.worker.Worker`.

The CUDA-specific behavior belongs behind the executor/runner pair:

- platform: `cuda`;
- runtime: start with the smallest working runtime, then move to
  persistent-device only when a serving kernel needs it;
- device selection: pass `device_id` or `device_ids` through executor kwargs;
- kernel selection: use explicit generated-kernel handles, not implicit global
  state.

## First Test Shape

The first local test should avoid a real model and assert serving plumbing:

1. Build a synthetic `RuntimeModel` with tiny tensors and a tokenizer fixture.
2. Instantiate `LLMEngine` with `SimplerNvExecutor`.
3. Call `generate()` with `max_new_tokens=1`.
4. Assert a deterministic token/text result.
5. Add an HTTP test for `/health` and `/v1/completions` only after the engine
   path works.

The first H200 test should be separate:

```bash
REMOTE_PTO_CU=<remote-pto-cu> \
  .agents/skills/cuda-backend-eval/scripts/run-remote-cuda.sh --sync -- \
  bash -lc 'source .venv/bin/activate && \
    PYTHONPATH=$PWD:$PWD/python \
    .venv/bin/python examples/cuda/<serving-shim-smoke>.py --require-cuda'
```

The actual example name should be chosen by the implementation PR.

## Non-Goals

- no DeepSeek-V4-Flash claim;
- no vLLM plugin claim;
- no throughput or latency claim;
- no FlashInfer integration claim;
- no RDMA, multi-node, or UCCL serving claim;
- no stable-doc promotion before the implementation PR is reviewed.

## Open Questions

- Which verified CUDA seed should the first shim call: a Gluon GEMM, a
  persistent MoE seed, or a purpose-built tiny logits kernel?
- Should `python/runtime/worker.py` grow CUDA-specific validation, or should the
  simpler-nv executor pass only generic worker kwargs?
- Should the first HTTP test live in this repo as a copied minimal harness, or
  in a future `pypto-serving` child branch?
