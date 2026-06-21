# CUDA Examples

These examples preserve review-facing CUDA backend metadata and provide small
skip-safe probes for current work. Historical benchmark rows do not run fresh
CUDA hardware checks; their A100/H200 measurements remain the `743709f3`
capture documented under `docs/nvidia-backend/history/`.

The FlashInfer serving operator checklist is recorded in
`docs/in_progress/nvidia_backend/flashinfer_serving_operator_checklist.md`.
It is a review guard for future attention, KV cache, sampling, GEMM, MoE,
FP8/FP4, and decode/prefill coverage, not integration evidence.

## Host-Schedule Vector Ops

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/host_schedule_vector_ops.py \
  --describe --op add --n 1024 --arch compute_80
```

Use `--op` to select the evaluated host-schedule ABI shape:
`add`, `mul`, `scale`, `square`, `axpy`, `affine`, `triad`, `quad`,
`generic_args`, or `generic_args4`.

## NCCL Two-GPU Baseline

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/nccl_two_gpu_baseline.py \
  --device-ids 0,1 --tensor-numel 1024 --require-cuda
```

This skip-safe probe establishes the NCCL compatibility floor for
`all_reduce`, `reduce_scatter`, `all_gather`, and `send_recv` through the
internal `simpler_setup.cuda_comm.CudaCommRuntimeRegistry` boundary. H200
evidence is recorded in
`docs/in_progress/nvidia_backend/nccl_two_h200_baseline.md`.

## NCCL Worker-Control Ops

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/nccl_worker_control_ops.py \
  --device-ids 6,7 --tensor-numel 1024 --build --require-cuda
```

This drives the same four float32 operations through descriptor-backed CUDA
host-runtime NCCL handles and the private `CTRL_COMM_OP` worker transport.
H200 evidence is recorded in
`docs/in_progress/nvidia_backend/nccl_worker_control_h200.md`.

## UCCL P2P IPC Adapter

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/uccl_p2p_ipc_adapter.py \
  --device-ids 0,1 --nbytes 1024 --require-cuda
```

This skip-safe probe maps a private `UcclP2PWriteIpcDescriptor` into the
Python-side `uccl.p2p` endpoint API when optional UCCL dependencies, CUDA, and
`torchrun` are available. It is adapter/probe evidence only, not CUDA
host-runtime UCCL dispatch, RDMA, multi-node, serving, or DeepSeek evidence.

## UCCL EP Dispatch/Combine Adapter

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/uccl_ep_dispatch_combine_adapter.py \
  --device-ids 0,1 --num-tokens 64 --hidden 128 \
  --num-topk 4 --num-experts 16 --input-dtype bf16 --require-cuda
```

This skip-safe probe maps `UcclEpDispatchCombineDescriptor` metadata into the
installed `uccl.ep` benchmark buffer when optional dependencies and
`UCCL_EP_BENCH_DIR` are available. It keeps UCCL-EP as Python-side
adapter/probe evidence and does not add a CUDA host-runtime UCCL ABI.

## pypto-serving simpler-nv Shim

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/pypto_serving_nv_shim.py \
  --prompt hello --max-new-tokens 2 --arch compute_90
```

This runs a synthetic `pypto-serving`-style `SimplerNvExecutor` and
`SimplerNvModelRunner` boundary for the simpler NVIDIA backend. It returns
deterministic `NV` tokens and skip-safe CUDA seed-launch evidence. H200
source-contract evidence is recorded in
`docs/in_progress/nvidia_backend/pypto_serving_source_contract_h200.md`.
The default launcher is the existing CUDA seed path. Use
`--kernel-launcher gluon-moe-expert` to route the same synthetic or cloned
source request through the generated Gluon MoE expert correctness harness for
`moe_expert_affine_f32`. Use
`--kernel-launcher persistent-moe-dispatch-combine` to route the same request
through the existing persistent-device MoE dispatch/combine example via
`run_moe_dispatch_combine(...)`.

Use `--openai-completion` to emit a synthetic `/v1/completions` response
shape, `--openai-chat-completion` to emit a synthetic
`/v1/chat/completions` response shape, `--engine` to route through the
synthetic `LLMEngine`-shaped fixture, `--http-fixture` to exercise the local
FastAPI fixture, or `--pypto-serving-source` to exercise the actual cloned
`pypto-serving` `create_serving_app` `/v1/completions` route with the
synthetic simpler-nv adapter. Use `--pypto-serving-source-chat` for the cloned
source `/v1/chat/completions` route. Use `--pypto-serving-source-stream` and
`--pypto-serving-source-chat-stream` for the cloned source `stream=true`
completion and chat routes.

Use `--pypto-serving-vllm-compat` to emit a JSON compatibility summary for
the four cloned source-route fixtures. The OpenAI-compatible structural fields
are: route, HTTP 200, object/model or stream shape,
text/message/delta presence, finish reason, non-streaming usage presence, and
streaming terminal `[DONE]` presence. It explicitly does not claim tokenizer
semantics, logprob values, stop-token semantics, production readiness,
throughput, latency, real DeepSeek weights, or simpler-nv/vLLM kernel
integration. With `--kernel-launcher persistent-moe-dispatch-combine`, the
aggregate command runs each source route in an isolated child process while
preserving per-route `pto_launch_results` metadata in the fixture summary.

The local FastAPI fixture exposes `/v1/completions` and
`/v1/chat/completions`. The chat fixture accepts a bounded non-streaming
OpenAI-style `messages` list with at least one user content entry and returns
a deterministic assistant message. The source-route chat fixture uses the
actual cloned `pypto-serving` server route with the same bounded message
shape and records review-safe route, status, assistant-message, `pto_status`,
and `pto_launch_count` fields. The source-route streaming fixtures summarize
review-safe SSE event counts, chunk counts, terminal `[DONE]`, assembled text
or assistant deltas, finish reason, PTO token IDs, launch count, and selected
kernel-launch metadata. Generated Gluon launch mode records
`launch_kind: gluon-moe-expert`, `kernel_name: moe_expert_affine_f32`, phase,
status, shape, generated artifact/source digest metadata, and numerical
max-error metadata when available. Persistent MoE launch mode records
`launch_kind: persistent-moe-dispatch-combine`,
`dag_shape: graph_descriptor_moe_dispatch_combine`, phase, status, shape,
completed count, max absolute error, scheduler error summary, and the Gluon
expert bridge/task-body digest when available.
It is not DeepSeek-V4-Flash serving, not vLLM plugin integration, not
FlashInfer integration, not production readiness, not throughput or latency,
not distributed serving, and not production fused MoE dispatch/combine serving
readiness.

## Persistent Layered-Cross Graph

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_layered_cross.py \
  --describe --n 1024 --arch compute_80 --scheduler-blocks 3
```

This describes the same `graph_descriptor_layered_cross` shape that feeds the
current `743709f3` benchmark gate.

## Persistent MoE Dispatch/Combine

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --output-json tmp/persistent-moe-dispatch-combine-local.json
```

This emits structured JSON for `graph_descriptor_moe_dispatch_combine`: four
expert transform tasks, one weighted combine task, and device-side fan-in
before the combine. Expert 0 uses the `gluon_gen` persistent task-body bridge
for `moe_expert_affine_f32` as func id `12`; the JSON includes
`gluon_expert_bridge` and a matching `task_bodies` entry for review. Without
CUDA tooling or a visible NVIDIA GPU it reports a skip; with `--require-cuda`,
the same skip returns a non-zero exit status.

For a bounded same-node two-device baseline, use:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --device-ids 6,7 --n 4096 --arch compute_90 --require-cuda
```

This runs the same persistent MoE graph independently on each requested CUDA
device in isolated child processes and aggregates validation for output error,
completion count, scheduler error state, fan-in state, and source/bridge
digests. It is same-node two-device baseline evidence, not fused cross-GPU
expert-parallel MoE, distributed serving, or performance evidence.

For the persistent MoE plus NCCL worker-control handoff gate, use:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --device-ids 6,7 --n 4096 --arch compute_90 \
  --with-nccl-handoff --tensor-numel 1024 --build --require-cuda
```

This first validates the same two-device persistent MoE aggregate, then runs
the descriptor-backed NCCL worker-control operations on the same device ids.
The JSON exposes persistent MoE validation, NCCL operation validation, tensor
size, output error, completion count, scheduler error state, and source/bridge
digests. It is not fused cross-GPU expert-parallel MoE, serving, RDMA,
multi-node, or performance evidence.

For the persistent MoE plus UCCL-EP adapter handoff gate, use:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/persistent_moe_dispatch_combine.py \
  --device-ids 6,7 --n 4096 --arch compute_90 \
  --with-uccl-ep-handoff --tensor-numel 1024 --require-cuda
```

This first validates the same two-device persistent MoE aggregate, then runs
the Python-side UCCL-EP dispatch/combine adapter on the same device ids. The
JSON exposes persistent MoE validation, UCCL-EP descriptor metadata, adapter
rank validation, tensor size, output error, completion count, scheduler error
state, and source/bridge digests. It is not fused cross-GPU expert-parallel
MoE, not CUDA host-runtime UCCL dispatch, and not serving, RDMA, multi-node,
or performance evidence.

## Gluon MoE Expert Affine

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_moe_expert_affine.py \
  --output-dir tmp/gluon-moe-expert-sweep-local \
  --arch compute_90 --sweep
```

This generates the `moe_expert_affine_f32` Gluon source and checks FP32
`out = scale_a * a + scale_b * b` vector correctness. The default command runs
one case; `--sweep` runs a fixed four-case shape and coefficient sweep with
aggregate structured JSON. Without CUDA tooling or a visible NVIDIA GPU it
reports skips; with `--require-cuda`, skipped cases return a non-zero exit
status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_moe_expert_h200.md`.

## Gluon Top-K Sampling

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_topk_sampling.py \
  --output-dir tmp/gluon-topk-sampling-local \
  --arch compute_90 --rows 3 --vocab 16 --k 5
```

This generates the `topk_sampling_f32` Gluon source and checks deterministic
top-k selection over FP32 logits fixtures. The default review fixture remains
`rows=2, vocab=8, k=3`; the broader fixture uses
`rows=3, vocab=16, k=5` with tied and negative logits. The JSON includes
shape, dtype, request metadata, CPU golden `values` and `indices`, GPU result
`values` and `indices` when CUDA runs, strict validation flags including
payload shape checks, repo-relative artifact paths, and explicit non-claims.
Without CUDA tooling or a visible NVIDIA GPU it reports a skip; with
`--require-cuda`, skipped cases return a non-zero exit status. H200 evidence
for the broader fixture is recorded in
`docs/in_progress/nvidia_backend/gluon_topk_sampling_h200.md`.
This is not FlashInfer integration evidence. It is also not vLLM or
simpler-nv kernel integration evidence, not DeepSeek serving correctness
evidence, not generated-text or tokenizer-semantics evidence, and not
throughput or latency evidence.

## Gluon Top-P Sampling

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_topp_sampling.py \
  --output-dir tmp/gluon-topp-sampling-local \
  --arch compute_90 --rows 3 --vocab 16 --max-k 6 --p 0.80
```

This generates the `topp_sampling_f32` Gluon source and checks deterministic
top-p selection over a small FP32 probability fixture whose rows already sum
to one. The default review fixture remains
`rows=2, vocab=8, max_k=5, p=0.75`; the broader fixture uses
`rows=3, vocab=16, max_k=6, p=0.80` with ties and rows that cross the
threshold at different selected counts. The JSON includes shape, dtype,
request metadata, CPU golden `values`, `indices`, `selected_counts`, the
effective cumulative probability boundary, GPU result fields when CUDA runs,
strict validation flags including payload shape checks, repo-relative
artifact paths, and explicit non-claims. Without CUDA tooling or a visible
NVIDIA GPU it reports a skip; with `--require-cuda`, skipped cases return a
non-zero exit status. H200 evidence for the broader fixture is recorded in
`docs/in_progress/nvidia_backend/gluon_topp_sampling_h200.md`.
This is not FlashInfer integration evidence. It is also not vLLM or
simpler-nv kernel integration evidence, not DeepSeek serving correctness
evidence, not generated-text or tokenizer-semantics evidence, and not
throughput or latency evidence.

## Gluon Min-P Sampling

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_minp_sampling.py \
  --output-dir tmp/gluon-minp-sampling-local \
  --arch compute_90
```

This generates the `minp_sampling_f32` Gluon source and checks deterministic
min-p selection over a small FP32 probability fixture whose rows already sum
to one. The fixed review shape is `rows=2, vocab=8, max_k=5, min_p=0.5`.
The operator selects candidates whose probability is at least `min_p` times
the row maximum, sorts selected candidates by probability descending with
lower token id first for ties, and fills unused output slots with `0.0`
values and `-1` indices. The JSON includes shape, dtype, request metadata,
CPU golden `values`, `indices`, `selected_counts`, GPU result fields when
CUDA runs, validation flags, repo-relative artifact paths, and explicit
non-claims. Without CUDA tooling or a visible NVIDIA GPU it reports a skip;
with `--require-cuda`, skipped cases return a non-zero exit status. H200
evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_minp_sampling_h200.md`.
This is not FlashInfer integration evidence. It is also not vLLM or
simpler-nv kernel integration evidence, not DeepSeek serving correctness
evidence, not generated-text or tokenizer-semantics evidence, and not
throughput or latency evidence.

## Gluon Speculative Decoding Accept/Reject

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_speculative_decoding.py \
  --output-dir tmp/gluon-speculative-decoding-local \
  --arch compute_90
```

This generates the `speculative_accept_f32` Gluon source and checks a
deterministic speculative-decoding accept/reject boundary over a small FP32
fixture. The fixed review shape is `rows=2, max_draft=4`. For each row,
draft token ids, draft probabilities, target probabilities for those same
draft tokens, and deterministic thresholds are provided directly. The
operator accepts while
`threshold <= min(1.0, target_probability / draft_probability)`, stops at
first reject per row, or put another way: stop at first reject and mask the
tail. It fills later output ids with `-1` and emits an integer accept mask
plus accepted counts. The JSON includes shape, dtype, request metadata, CPU
golden `accepted_token_ids`, `accept_mask`, and
`accepted_counts`, GPU result fields when CUDA runs, validation flags,
repo-relative artifact paths, and explicit non-claims. Without CUDA tooling
or a visible NVIDIA GPU it reports a skip; with `--require-cuda`, skipped
cases return a non-zero exit status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_speculative_decoding_h200.md`.
This is not FlashInfer integration evidence. It is also not vLLM or
simpler-nv kernel integration evidence, not DeepSeek serving correctness
evidence, not generated-text or tokenizer-semantics evidence, and not
throughput or latency evidence.

## Gluon FP32 FlashAttention

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_flashattention_fwd.py \
  --output-dir tmp/gluon-flashattention-local \
  --arch compute_90 --sweep
```

This generates the `flashattention_fwd_f32` Gluon source and checks
`softmax((q @ k.T) * scale) @ v` correctness. The default command runs one
bounded `32x32x32` single-tile shape; `--sweep` runs an aggregate structured
JSON shape sweep with the existing `32x32x32` case and a bounded
`head_dim=64` case selected after `32x32x64 failed H200 correctness`. The
stdout JSON includes `schema_version`, aggregate status, per-case provenance,
repo-relative artifact paths, and sanitized error text. Without CUDA tooling,
a visible NVIDIA GPU, or Gluon `gl.dot_fma`, it reports skips; with
`--require-cuda`, skipped or failed cases return a non-zero exit status. H200
evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_flashattention_h200.md`.

## Gluon FP32 RMSNorm

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_rmsnorm_f32.py \
  --output-dir tmp/gluon-rmsnorm-shape-coverage-local \
  --arch compute_90 --sweep
```

This generates the `rmsnorm_f32` Gluon source and checks FP32
`x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight`
correctness. The default command still runs one case and accepts
`--rows 2 --hidden 16 --eps 1e-5`; `--sweep` runs a fixed two-case shape
sweep covering `hidden=16` from the existing smoke case and `hidden=7168`
from `DeepSeek-V4-Flash config hidden_size` provenance. Without CUDA tooling
or a visible NVIDIA GPU it reports skips; with `--require-cuda`, skipped cases
return a non-zero exit status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_rmsnorm_h200.md`.

## Gluon FP32 Gemma Fused RMSNorm

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_gemma_fused_rmsnorm_f32.py \
  --output-dir tmp/gluon-gemma-fused-rmsnorm-local \
  --rows 2 --hidden 16 --eps 1e-5 --arch compute_90
```

This generates the `gemma_fused_rmsnorm_f32` Gluon source and checks FP32
`out[row, col] = x[row, col] * rsqrt(mean(x[row, :]^2) + eps) * (1.0 + weight[col])`
correctness for one bounded Gemma-style fused normalization shape. Without
CUDA tooling or a visible NVIDIA GPU it reports a skip; with `--require-cuda`,
the same skip returns a non-zero exit status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_gemma_fused_rmsnorm_h200.md`.

## Gluon FP32 LayerNorm

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_layernorm_f32.py \
  --output-dir tmp/gluon-layernorm-shape-coverage-local \
  --arch compute_90 --sweep
```

This generates the `layernorm_f32` Gluon source and checks FP32
`out = (x - mean) * rsqrt(var + eps) * weight + bias` correctness with
per-hidden weight and bias. The default command still runs one case and
accepts `--rows 2 --hidden 16 --eps 1e-5`; `--sweep` runs a fixed two-case
shape sweep covering `hidden=16` from the existing smoke case and
`hidden=7168` from `DeepSeek-V4-Flash config hidden_size` provenance. Without
CUDA tooling or a visible NVIDIA GPU it reports skips; with `--require-cuda`,
skipped cases return a non-zero exit status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_layernorm_h200.md`.

## Gluon FP32 RoPE

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_rope_f32.py \
  --output-dir tmp/gluon-rope-shape-coverage-local \
  --arch compute_90 --sweep
```

This generates the `rope_f32` Gluon source and checks FP32 rotary position
embedding math over adjacent even/odd feature pairs. The default command still
runs one case and accepts `--batch 1 --seq 2 --head-dim 8`; `--sweep` runs
aggregate structured JSON covering the existing smoke case and a bounded
`batch=1, seq=4, head_dim=64` case with
`tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
`rope_head_dim: 64` provenance. Without CUDA tooling or a visible NVIDIA GPU
it reports skips; with `--require-cuda`, skipped or failed cases return a
non-zero exit status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_rope_h200.md`.

## Gluon FP32 SiLU

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_silu_f32.py \
  --output-dir tmp/gluon-silu-shape-coverage-local \
  --arch compute_90 --sweep
```

This generates the `silu_f32` Gluon source and checks FP32
`out = x / (1.0 + exp(-x))` correctness. The default command still runs one
case and accepts `--n 32`; `--sweep` runs aggregate structured JSON covering
the existing smoke case and a bounded `n=2048` case with
`tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
`moe_inter_dim: 2048` and `swiglu_limit: 10.0` provenance as standalone SiLU
gate-activation-width evidence. Without CUDA tooling or a visible NVIDIA GPU
it reports skips; with `--require-cuda`, skipped or failed cases return a
non-zero exit status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_silu_h200.md`.

## Gluon FP32 GELU

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_gelu_f32.py \
  --output-dir tmp/gluon-gelu-shape-coverage-local \
  --sweep --arch compute_90
```

This generates the `gelu_f32` Gluon source and checks FP32
`0.5 * x * (1.0 + erf(x / sqrt(2.0)))` correctness for a small fixed sweep:
the existing `n=32` smoke case and a bounded `n=2048` case with
`tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
`moe_inter_dim: 2048` and `swiglu_limit: 10.0` provenance as standalone GELU
activation-width evidence. Without CUDA tooling, a visible NVIDIA GPU, or
Gluon `gl.erf`, it reports skips; with `--require-cuda`, skipped or failed
cases return a non-zero exit status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_gelu_h200.md`.

## Gluon FP32 Gated SiLU

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_gated_silu_f32.py \
  --output-dir tmp/gluon-gated-silu-local \
  --n 32 --arch compute_90
```

This generates the `gated_silu_f32` Gluon source and checks FP32
`out = value * gate / (1.0 + exp(-gate))` correctness for one bounded vector
shape. Without CUDA tooling, a visible NVIDIA GPU, or Gluon `gl.exp`, it
reports a skip; with `--require-cuda`, the same skip returns a non-zero exit
status. H200 evidence is recorded in
`docs/in_progress/nvidia_backend/gluon_gated_silu_h200.md`.

The review sweep keeps the default single-case behavior available while adding
the existing `n=32` smoke case and a bounded `n=2048` case with
`tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
`moe_inter_dim: 2048` and `swiglu_limit: 10.0` provenance:

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/gluon_gated_silu_f32.py \
  --output-dir tmp/gluon-gated-silu-shape-coverage-local \
  --sweep --arch compute_90
```

## DeepSeek V4 Flash Weight Manifest

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/deepseek_v4_flash_weight_manifest.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --metadata tmp/sources/model-metadata/deepseek-ai-DeepSeek-V4-Flash.json \
  --require-complete
```

This checks local gitignored shard presence against
`model.safetensors.index.json`. It also reports a preflight capacity surface:
`required_missing_bytes`, `storage_free_bytes`, `storage_required_bytes`,
`storage_has_capacity`, `preflight_status`, `next_gate`, and the later
model-load `next_command`. Pass `--storage-dir <repo-relative tmp path>` to
check a selected non-committed artifact directory, or
`--require-preflight` to exit `3` unless the manifest permits the model-load
gate. The local preflight evidence is recorded in
`docs/in_progress/nvidia_backend/deepseek_v4_flash_weight_manifest_preflight.md`
note.
The completed local artifact evidence is recorded in
`docs/in_progress/nvidia_backend/deepseek_v4_flash_weight_manifest_complete.md`.
It is not model-load or serving evidence.

## DeepSeek V4 Flash Artifact Probe

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv/bin/python examples/cuda/vllm_deepseek_v4_artifact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash
```

This combines local config/tokenizer/index/shard readiness with the existing
weight-free vLLM DeepSeek V4 import and synthetic config probes. Missing local
artifacts or missing vLLM report structured skips by default; use
`--require-artifacts` or `--require-vllm` to make either condition fail the
command. It does not attempt model load, start a server, or run inference.

The remote H200 readiness slices are recorded in
`docs/in_progress/nvidia_backend/vllm_remote_install_probe.md` and
`docs/in_progress/nvidia_backend/deepseek_v4_flash_serving_readiness.md`.
They record remote H200 reachability and the current serving-readiness
boundary.
The follow-up environment/artifact gate is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_env_artifact_probe.md`: the
remote vLLM import/config probes pass in `.venv-vllm-probe`, while the
artifact gates fail because the repo-relative artifact path contains
metadata/tokenizer files but not the indexed weight shards. The artifact
completion gate is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_artifact_complete.md`: the same
remote vLLM import/config probes pass, and the artifact/manifest gates now
find all 46 indexed shards at the repo-relative artifact path. These gates are
not model-load or serving evidence.

## DeepSeek V4 Flash vLLM Model-Load Probe

```bash
PYTHONPATH=$PWD:$PWD/python \
  .venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_model_load_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --require-artifacts --require-vllm --require-cuda \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager
```

Run it only under an explicit GPU boundary, for example
`CUDA_VISIBLE_DEVICES=<two ids>` with a matching `--tensor-parallel-size 2`,
and an external timeout.

Missing artifacts, missing vLLM, or missing CUDA report structured skips by
default; the matching `--require-*` flags convert those preflight skips into
failures. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_model_load_probe.md`: vLLM loaded
all 46 shards and initialized an `LLMEngine` on two H200 GPUs at
`max_model_len=4096`. This is model-load and engine-initialization evidence,
not server health, inference correctness, 256K context, throughput, latency, or
production-readiness evidence.

## DeepSeek V4 Flash vLLM Server Health Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28123 \
  --server-log tmp/vllm-server-health-probe/server-28123.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This starts a local-only OpenAI-compatible vLLM server bound to `127.0.0.1`,
checks `/health` and `/v1/models`, emits structured JSON, and terminates the
server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_server_health_probe.md`: the
server started for `deepseek-ai/DeepSeek-V4-Flash`, returned HTTP 200 from
both checked endpoints, and shut down with no remaining process-group PIDs
reported by the probe. This is server startup and health/model-list evidence,
not generated-text correctness, tokenizer semantics, 256K context,
throughput, latency, production readiness, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM 64K Context Health Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28132 \
  --server-log tmp/vllm-64k-context-health-probe/server-28132.log \
  --max-model-len 65536 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This reuses the local-only server-health probe with a larger
`--max-model-len` and still checks only `/health` and `/v1/models`; it does
not send a long prompt or run generation. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_64k_context_health_probe.md`:
the server returned HTTP 200 from both checked endpoints and the model list
reported `max_model_len=65536`. This is a 64K server-health/model-list
capacity gate only, not generated-text correctness, tokenizer semantics,
prompt correctness, 256K context, throughput, latency, production readiness,
or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 128K Context Health Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28133 \
  --server-log tmp/vllm-128k-context-health-probe/server-28133.log \
  --max-model-len 131072 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This reuses the local-only server-health probe with a larger
`--max-model-len` and still checks only `/health` and `/v1/models`; it does
not send a long prompt or run generation. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_128k_context_health_probe.md`:
the server returned HTTP 200 from both checked endpoints and the model list
reported `max_model_len=131072`. This is a 128K server-health/model-list
capacity gate only, not generated-text correctness, tokenizer semantics,
prompt correctness, 256K context, throughput, latency, production readiness,
or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Context Health Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 50m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_server_health_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28134 \
  --server-log tmp/vllm-256k-context-health-probe/server-28134.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --terminate-timeout-seconds 60
```

This reuses the local-only server-health probe with the model's configured
262,144-token length and still checks only `/health` and `/v1/models`; it does
not send a long prompt or run generation. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_256k_context_health_probe.md`:
the server returned HTTP 200 from both checked endpoints and the model list
reported `max_model_len=262144`. This is a 256K server-health/model-list
capacity gate only, not long-prompt behavior, generated-text correctness,
tokenizer semantics, prompt correctness, throughput, latency, production
readiness, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Long-Prompt Admission Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_admission_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28135 \
  --server-log tmp/vllm-long-prompt-admission-probe/server-28135.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 1 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 16K prompt-token budget, records only
review-safe shape/accounting fields, and terminates the server process group.
The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_long_prompt_admission_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=15995`, and cleanup reported no remaining process-group PIDs.
This is long-prompt admission evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 70m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28136 \
  --server-log tmp/vllm-long-prompt-response-contract-probe/server-28136.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 16K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=15995`, `completion_tokens=4`, and `total_tokens=15999`, and
cleanup reported no remaining process-group PIDs. This is long-prompt
response-contract evidence only, not generated-text correctness, tokenizer
semantic correctness, prompt semantic correctness, token identity, logprob,
stop-token, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Chat Exact Canary Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_exact_canary_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28149 \
  --server-log tmp/vllm-chat-exact-canary-probe/server-28149.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --max-tokens 16 --temperature 0.0 --top-p 1.0 \
  --seed 0 --expected-answer PTO_CHAT_EXACT_CANARY_28149
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/chat/completions` request,
and requires the narrowly normalized assistant content to exactly equal the
expected canary string. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_exact_canary_probe.md`.
This is a bounded OpenAI-compatible chat-completions exact-output canary, not
general generated-text correctness, semantic correctness, long-prompt chat
behavior, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Chat 256K Needle Exact Probe

````bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_256k_needle_exact_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28151 \
  --server-log tmp/vllm-chat-256k-needle-exact-probe/server-28151.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151 \
  --stop-sequence $'\n```'
````

This starts the same local-only two-H200 vLLM server boundary, checks
`/health` and `/v1/models`, sends one bounded non-streaming
`/v1/chat/completions` request with a synthetic needle placed inside a
near-256K user entry, and requires the narrowly normalized assistant content
to exactly equal `PTO_CHAT_NEEDLE_256K_CONTEXT_OK_28151`. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_exact_probe.md`:
the request returned HTTP 200, `finish_reason=stop`, usage reported
`prompt_tokens=255795`, `completion_tokens=18`, and `total_tokens=255813`,
the strict exact check passed, and cleanup reported no remaining
process-group PIDs. This is one synthetic chat-completions needle exact-output
gate, not general generated-text correctness, semantic correctness,
throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Chat 256K Needle Repeat Probe

````bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_256k_needle_repeat_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28152 \
  --server-log tmp/vllm-chat-256k-needle-repeat-probe/server-28152.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152
````

This starts the same local-only two-H200 vLLM server boundary, checks
`/health` and `/v1/models`, sends exactly two identical bounded
non-streaming `/v1/chat/completions` requests with a synthetic needle placed
inside a near-256K user entry, and requires the narrowly normalized assistant
content from both attempts to exactly equal
`PTO_CHAT_NEEDLE_256K_REPEAT_OK_28152`. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_repeat_probe.md`:
both requests returned HTTP 200, `finish_reason=stop`, usage reported
`prompt_tokens=255796`, `completion_tokens=19`, and `total_tokens=255815`,
both strict exact checks passed, and cleanup reported no remaining
process-group PIDs. This is one two-request synthetic chat-completions needle
exact-repeat gate, not general generated-text correctness, semantic
correctness, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Chat 256K Needle Streaming Probe

````bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28153 \
  --server-log tmp/vllm-chat-256k-needle-stream-probe/server-28153.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_OK_28153
````

This starts the same local-only two-H200 vLLM server boundary, checks
`/health` and `/v1/models`, sends one bounded streaming
`/v1/chat/completions` request with a synthetic needle placed inside a
near-256K user entry, parses server-sent events, and requires the narrowly
normalized assembled assistant content to exactly equal
`PTO_CHAT_NEEDLE_256K_STREAM_OK_28153`. The remote H200 evidence is recorded
in
`docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_probe.md`:
the request returned HTTP 200, parsed 19 SSE events and 16 assistant content
deltas, `finish_reason=stop`, usage was `not_returned`, the strict exact
check passed, and cleanup reported no remaining process-group PIDs. This is
one synthetic streaming chat-completions needle exact-output gate, not
general generated-text correctness, semantic correctness, throughput, latency,
production readiness, broad determinism, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM Chat 256K Needle Streaming Repeat Probe

````bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_repeat_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28154 \
  --server-log tmp/vllm-chat-256k-needle-stream-repeat-probe/server-28154.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_REPEAT_OK_28154 \
  --repeat-count 2
````

This starts the same local-only two-H200 vLLM server boundary, checks
`/health` and `/v1/models`, sends exactly two identical bounded streaming
`/v1/chat/completions` requests with a synthetic needle placed inside a
near-256K user entry, parses server-sent events for both attempts, and
requires each attempt to receive terminal `[DONE]`, record a final
`finish_reason`, and narrowly normalize to exactly
`PTO_CHAT_NEEDLE_256K_STREAM_REPEAT_OK_28154`. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_repeat_probe.md`.
Both attempts returned HTTP 200, each parsed 22 SSE events and 19 assistant
content deltas, each received terminal `[DONE]`, each reported
`finish_reason=stop`, usage was `not_returned`, both strict exact checks
passed, and cleanup reported no remaining process-group PIDs. This is one
synthetic streaming chat-completions needle exact-repeat gate, not general
generated-text correctness, semantic correctness, throughput, latency,
production readiness, broad determinism, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM Chat 256K Needle Streaming Position Sweep Probe

````bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_position_sweep_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28156 \
  --server-log tmp/vllm-chat-256k-needle-stream-position-sweep-probe/server-28156.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156 \
  --needle-position-sweep early,middle,late
````

This starts the same local-only two-H200 vLLM server boundary, checks
`/health` and `/v1/models`, sends exactly one bounded streaming
`/v1/chat/completions` request per requested synthetic needle position, parses
server-sent events for each position, and requires every position to receive
terminal `[DONE]`, record a final `finish_reason`, and narrowly normalize to
exactly `PTO_CHAT_NEEDLE_256K_STREAM_SWEEP_OK_28156`. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_position_sweep_probe.md`.
The early and middle requests each parsed 22 SSE events, and the late request
parsed 21 SSE events. Each position assembled 19 assistant content deltas,
received terminal `[DONE]`, reported `finish_reason=stop`, returned usage as
`not_returned`, passed strict exact matching, and cleanup reported no
remaining process-group PIDs. This is one synthetic streaming
chat-completions needle position-sweep gate, not general generated-text
correctness, semantic correctness, throughput, latency, production readiness,
broad determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Chat 256K Needle Streaming Usage Contract Probe

````bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_usage_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28159 \
  --server-log tmp/vllm-chat-256k-needle-stream-usage-token-accounting/server-28159.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28159
````

This starts the same local-only two-H200 vLLM server boundary, checks
`/health` and `/v1/models`, calls `/tokenize` for the same chat messages,
sends one bounded streaming `/v1/chat/completions` request with
`stream_options.include_usage=true`, and requires both returned streaming
usage and strict exact output matching. The narrowly normalized assembled
assistant content must exactly equal
`PTO_CHAT_NEEDLE_256K_STREAM_USAGE_OK_28159`, and the returned streaming usage
object must satisfy the prompt-token, completion-token, and total-token
contract checks against the server-side `/tokenize` count. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_usage_contract_probe.md`.
The server-tokenize accounting rerun returned HTTP 200 from `/health`,
`/v1/models`, `/tokenize`, and `/v1/chat/completions`, parsed 22 JSON SSE
events, saw 18 assistant content deltas, received terminal `[DONE]`, recorded
`finish_reason=stop`, passed strict exact output matching, and returned
streaming usage keys in a final usage-bearing event with `choice_count=0`.
The gate passed with `PROBE_EXIT_STATUS=0`: `/tokenize` measured
`prompt_tokens=255797`, streaming usage reported `usage.prompt_tokens=255797`,
`usage.completion_tokens=20` was within `max_tokens=64`, and
`usage.total_tokens=255817` passed the total-token bound. This is one
synthetic streaming chat-completions needle usage-contract pass, not general
generated-text correctness, semantic correctness, throughput, latency,
production readiness, broad determinism, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM Chat 256K Needle Streaming Truncated Failure Probe

````bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_256k_needle_stream_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28155 \
  --server-log tmp/vllm-chat-256k-needle-stream-truncated-failure-probe/server-28155.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 1 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_CHAT_NEEDLE_256K_STREAM_TRUNCATED_OK_28155 \
  --stop-sequence $'\n```'
````

This reuses the existing streaming chat 256K needle probe under the same
local-only two-H200 vLLM server boundary, but intentionally gives the strict
exact comparator an insufficient one-token completion budget. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_256k_needle_stream_truncated_failure_probe.md`:
the streaming request returned HTTP 200, parsed SSE events, received terminal
`[DONE]`, reported `finish_reason=length`, failed strict exact mode with
`chat_needle_stream_expected_answer_not_exact`, and cleanup reported no
remaining process-group PIDs. This is an expected failure-mode
characterization, not a transport/server failure, generated-text correctness,
semantic correctness, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Chat Exact Truncated Failure Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_chat_exact_canary_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28150 \
  --server-log tmp/vllm-chat-exact-truncated-failure-probe/server-28150.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --max-tokens 1 --temperature 0.0 --top-p 1.0 \
  --seed 0 --expected-answer PTO_CHAT_EXACT_CANARY_28149
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/chat/completions` request,
and intentionally gives the strict exact comparator an insufficient
one-token generation budget. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_chat_exact_truncated_failure_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=33`, `completion_tokens=1`, and `total_tokens=34`, strict exact
mode failed with `chat_canary_expected_answer_not_exact`, and cleanup
reported no remaining process-group PIDs. This is an expected failure-mode
characterization, not generated-text correctness, semantic correctness,
long-prompt chat behavior, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Long-Prompt Warmup/Follow-Up Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 75m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_warmup_followup_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28137 \
  --server-log tmp/vllm-long-prompt-warmup-followup-probe/server-28137.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 16000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --log-settle-seconds 2
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one labeled warmup synthetic
non-streaming `/v1/completions` request near a 16K prompt-token budget, sends
one labeled follow-up request with the same request shape, validates
review-safe response-contract fields for both responses, records generated
text lengths only, and terminates the server process group. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_long_prompt_warmup_followup_probe.md`:
both requests returned HTTP 200 with one response choice, both responses
reported `prompt_tokens=15995`, `completion_tokens=4`, and
`total_tokens=15999`, and cleanup reported no remaining process-group PIDs.
This is long-prompt warmup/follow-up response-contract evidence only, not
generated-text correctness, tokenizer semantic correctness, prompt semantic
correctness, token identity, logprob, stop-token, throughput, latency,
production readiness, broad determinism, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM 32K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 80m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28138 \
  --server-log tmp/vllm-32k-long-prompt-response-contract-probe/server-28138.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 32000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 32K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_32k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=32000`, `completion_tokens=4`, and `total_tokens=32004`, and
cleanup reported no remaining process-group PIDs. This is 32K long-prompt
response-contract evidence only, not generated-text correctness, tokenizer
semantic correctness, prompt semantic correctness, token identity, logprob,
stop-token, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 64K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 90m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28139 \
  --server-log tmp/vllm-64k-long-prompt-response-contract-probe/server-28139.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 600 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 64000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 64K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_64k_long_prompt_response_contract_probe.md`:
the request
returned HTTP 200 with one response choice, usage reported
`prompt_tokens=63999`, `completion_tokens=4`, and `total_tokens=64003`, and
cleanup reported no remaining process-group PIDs. This is 64K long-prompt
response-contract evidence only, not generated-text correctness, tokenizer
semantic correctness, prompt semantic correctness, token identity, logprob,
stop-token, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 128K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28140 \
  --server-log tmp/vllm-128k-long-prompt-response-contract-probe/server-28140.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 900 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 128000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 128K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_128k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=127997`, `completion_tokens=4`, and `total_tokens=128001`,
and cleanup reported no remaining process-group PIDs. This is 128K
long-prompt response-contract evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 192K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28141 \
  --server-log tmp/vllm-192k-long-prompt-response-contract-probe/server-28141.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1200 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 192000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 192K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_192k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=191995`, `completion_tokens=4`, and `total_tokens=191999`,
and cleanup reported no remaining process-group PIDs. This is 192K
long-prompt response-contract evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Long-Prompt Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_long_prompt_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28142 \
  --server-log tmp/vllm-256k-long-prompt-response-contract-probe/server-28142.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 256000 --max-tokens 4 \
  --temperature 0.0 --top-p 1.0 --seed 0
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 256K prompt-token budget with `max_tokens=4`,
validates review-safe response-contract fields, records generated text length
only, and terminates the server process group. The remote H200 evidence is
recorded in
`docs/in_progress/nvidia_backend/vllm_remote_256k_long_prompt_response_contract_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=256004`, `completion_tokens=4`, and `total_tokens=256008`,
and cleanup reported no remaining process-group PIDs. This is 256K
long-prompt response-contract evidence only, not generated-text correctness,
tokenizer semantic correctness, prompt semantic correctness, token identity,
logprob, stop-token, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Needle Correctness Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28143 \
  --server-log tmp/vllm-256k-needle-correctness-probe/server-28143.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143 \
  --match-mode contains
```

This starts a local-only vLLM server bound to `127.0.0.1` with the
262,144-token model length, sends one synthetic non-streaming
`/v1/completions` request near a 256K prompt-token budget, and reports
`status=passed` only when the short generated output satisfies the selected
match mode. The default `contains` mode preserves the existing containment
gate. The remote H200 containment evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_256k_needle_correctness_probe.md`:
the request returned HTTP 200 with one response choice, usage reported
`prompt_tokens=255799`, `completion_tokens=64`, and `total_tokens=255863`,
the generated output contained `PTO_NEEDLE_256K_CONTEXT_OK_28143`, and cleanup
reported no remaining process-group PIDs. This is synthetic needle retrieval
correctness evidence only, not general generated-text correctness, semantic
correctness, throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM 256K Needle Exact-Output Probe

```bash
CUDA_VISIBLE_DEVICES=1,7 VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 120m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_needle_correctness_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28144 \
  --server-log tmp/vllm-256k-needle-exact-output-probe/server-28144.log \
  --max-model-len 262144 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 1800 --terminate-timeout-seconds 60 \
  --target-prompt-tokens 255800 --max-tokens 64 \
  --temperature 0.0 --top-p 1.0 --seed 0 \
  --expected-answer PTO_NEEDLE_256K_CONTEXT_OK_28143 \
  --match-mode exact
```

Exact mode strips leading/trailing whitespace, strips one surrounding
Markdown code fence only when the whole output is fenced, and then compares
the normalized generated output exactly to the expected synthetic answer. It
does not remove explanatory sentences, punctuation, unmatched Markdown fences,
or unrelated tokens. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_256k_needle_exact_output_failure.md`:
the request returned HTTP 200 with one response choice, but strict exact mode
failed because the normalized output retained an unmatched closing Markdown
fence after `PTO_NEEDLE_256K_CONTEXT_OK_28143`. This is a strict synthetic
exact-output failure record only, not general generated-text correctness,
semantic correctness, throughput, latency, production readiness, broad
determinism, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Inference Smoke Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 55m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_inference_smoke_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28124 \
  --server-log tmp/vllm-inference-smoke-probe/server-28124.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --endpoint /v1/completions --prompt Hello --max-tokens 1 \
  --temperature 0.0
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends exactly one bounded completion request by default, records
request limits and response shape, and terminates the server process group.
The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_inference_smoke_probe.md`: the
server returned HTTP 200 from readiness endpoints, returned HTTP 200 from one
`/v1/completions` request with `max_tokens=1`, and shut down with no remaining
process-group PIDs reported by the probe. This is inference-smoke evidence,
not generated-text correctness, tokenizer semantics, prompt correctness, 256K
context, throughput, latency, production readiness, or simpler-nv/vLLM
integration evidence.

## DeepSeek V4 Flash vLLM Response-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 60m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28125 \
  --server-log tmp/vllm-response-contract-probe/server-28125.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 --seed 0
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming completion request with explicit
sampler settings, validates OpenAI-compatible response structure, and
terminates the server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_response_contract_probe.md`: the
server returned HTTP 200 from readiness endpoints and `/v1/completions`, the
response had exactly one choice, usage token counts were internally
consistent and within the request bound, and cleanup reported no remaining
process-group PIDs. This is response-contract evidence, not generated-text
correctness, tokenizer semantics, prompt correctness, 256K context,
throughput, latency, production readiness, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM Warmup-Shape Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_warmup_shape_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28126 \
  --server-log tmp/vllm-warmup-shape-probe/server-28126.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --log-settle-seconds 2
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one labeled warmup `/v1/completions` request, sends one
follow-up same-shape `/v1/completions` request, validates the same structural
response contract for both responses, counts selected Triton JIT warning
strings in server-log windows around the two requests, and terminates the
server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_warmup_shape_probe.md`. This is
warmup-shape observation evidence, not generated-text correctness, tokenizer
semantics, prompt correctness, 256K context, throughput, latency, production
readiness, warmup-eliminates-JIT evidence, or simpler-nv/vLLM integration
evidence.

## DeepSeek V4 Flash vLLM Request-Shape Variation Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_request_shape_variation_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28127 \
  --server-log tmp/vllm-request-shape-variation-probe/server-28127.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --variation-max-tokens 8 --log-settle-seconds 2
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one known warmup `/v1/completions` request, sends one
same-shape follow-up request to preserve the prior baseline inside the run,
sends one bounded variation request with a different prompt and
`max_tokens=8`, validates the same structural response contract for all three
responses, counts selected Triton JIT warning strings by request window, and
terminates the server process group. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_request_shape_variation_probe.md`.
This is request-shape variation observation evidence, not generated-text
correctness, tokenizer semantics, prompt correctness, 256K context,
throughput, latency, production readiness, broad warmup-eliminates-JIT
evidence, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Serving-Semantics Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_serving_semantics_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28128 \
  --server-log tmp/vllm-serving-semantics-probe/server-28128.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --max-tokens 8 --temperature 0.0 --top-p 1.0 \
  --seed 0 --log-settle-seconds 2
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends two identical bounded deterministic `/v1/completions`
requests, validates the structural response contract for both responses,
then compares response observations: completion text digest, text length,
`finish_reason`, and usage accounting. The generated text is not recorded or
judged for correctness. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_serving_semantics_probe.md`.
This is a bounded serving-semantics observation, not generated-text
correctness, tokenizer semantics, prompt correctness, 256K context,
throughput, latency, production readiness, broad determinism evidence, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Logprobs-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28129 \
  --server-log tmp/vllm-logprobs-contract-probe/server-28129.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --logprobs-contract --logprobs 1 --prompt-logprobs 1
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/completions` request with
explicit `logprobs=1` and `prompt_logprobs=1`, validates the base structural
response contract, then checks only logprob response shape: completion
logprob list lengths match `usage.completion_tokens`, and
`choice.prompt_logprobs` is list-valued and bounded by
`usage.prompt_tokens`. The remote H200 evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_logprobs_contract_probe.md`.
This is a bounded logprobs response-shape observation, not generated-text
correctness, tokenizer semantics, prompt correctness, token identity or
logprob value correctness, 256K context, throughput, latency, production
readiness, or simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Echo-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28130 \
  --server-log tmp/vllm-echo-contract-probe/server-28130.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 1 --temperature 0.0 --top-p 1.0 \
  --seed 0 --echo-contract
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/completions` request with
explicit `echo=true`, validates the base structural response contract, then
checks only echo response shape: the request prompt is string-valued and the
response text starts with that prompt. The remote H200 evidence is recorded
in `docs/in_progress/nvidia_backend/vllm_remote_echo_contract_probe.md`. This
is a bounded echo response-shape observation, not generated-text correctness,
tokenizer semantics, prompt correctness, token identity or logprob value
correctness, 256K context, throughput, latency, production readiness, or
simpler-nv/vLLM integration evidence.

## DeepSeek V4 Flash vLLM Stop-Contract Probe

```bash
CUDA_VISIBLE_DEVICES=<two ids> VLLM_NO_USAGE_STATS=1 \
PYTHONPATH=$PWD:$PWD/python \
timeout --foreground 65m \
.venv-vllm-probe/bin/python \
  examples/cuda/vllm_deepseek_v4_response_contract_probe.py \
  --artifact-dir tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash \
  --vllm-bin .venv-vllm-probe/bin/vllm \
  --port 28131 \
  --server-log tmp/vllm-stop-contract-probe/server-28131.log \
  --max-model-len 4096 --tensor-parallel-size 2 \
  --dtype bfloat16 --quantization deepseek_v4_fp8 \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 \
  --distributed-executor-backend mp --enforce-eager \
  --timeout-seconds 2700 --poll-interval-seconds 10 \
  --request-timeout-seconds 180 --terminate-timeout-seconds 60 \
  --prompt Hello --max-tokens 4 --temperature 0.0 --top-p 1.0 \
  --seed 0 --stop-contract
```

This starts the same local-only server boundary, checks `/health` and
`/v1/models`, sends one bounded non-streaming `/v1/completions` request with
explicit `stop`, `stop_token_ids`, and `include_stop_str_in_output=false`,
validates the base structural response contract, and checks only that the
explicit stop fields were carried in the accepted request. The remote H200
evidence is recorded in
`docs/in_progress/nvidia_backend/vllm_remote_stop_contract_probe.md`. This is
bounded stop-field acceptance and response-contract evidence, not stop-trigger
evidence, generated-text correctness, tokenizer semantics, prompt
correctness, token identity or stop-token semantic correctness, 256K context,
throughput, latency, production readiness, broad determinism, or
simpler-nv/vLLM integration evidence.
