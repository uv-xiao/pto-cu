# FlashInfer Serving Operator Checklist

This checklist turns the current FlashInfer source read into a PTO serving
operator review guard. It is a planning artifact only: it records the operator
families PTO should cover before future serving-readiness claims.

## Sources Read

- FlashInfer README:
  `tmp/sources/repos/external/flashinfer/README.md`. The finding used is that
  the README names serving attention, KV cache, GEMM, MoE, sampling,
  communication, RoPE, normalization, activation, low-precision, and
  GPU-support families.

The README says FlashInfer is a library and kernel generator for inference
with unified APIs for attention, GEMM, and MoE. It also lists low-precision
FP8 and FP4 scope for attention, GEMM, and MoE operations.

Serving-relevant families verified from the README:

- attention kernels: Paged and Ragged KV-Cache, Decode, Prefill, and Append,
  MLA Attention, Cascade Attention, Sparse Attention, and POD-Attention;
- GEMM and linear operations: BF16 GEMM, FP8 GEMM, FP4 GEMM, and Grouped GEMM;
- MoE: Fused MoE Kernels, Multiple Routing Methods, and Quantized MoE;
- sampling and decoding: Sorting-Free Sampling and Speculative Decoding;
- communication: AllReduce, Multi-Node NVLink, and NVSHMEM Integration;
- other operators: RoPE, Normalization, and Activations;
- hardware support: Hopper SM 9.0 includes H100 and H200.

## Checklist

### Attention Kernels

- **FlashInfer reference family:** attention kernels: Paged and Ragged
  KV-Cache; Decode, Prefill, and Append; MLA Attention; Cascade Attention;
  Sparse Attention; POD-Attention.
- **PTO current evidence:** `gluon_flashattention_h200.md` records a small
  FP32 FlashAttention shape sweep. It includes `schema_version`, aggregate
  status, per-case provenance, repo-relative artifact paths, and an explicit
  `--tile-shape 32x32x64` H200 single-case repro. The sweep keeps the existing
  `32x32x32` case and a bounded `head_dim=64` case selected after `32x32x64
  failed H200 correctness`; the `32x32x64` repro now passes with structured
  JSON and remains separate from the promoted sweep. vLLM DeepSeek probes are
  real vLLM serving evidence and mention fp8 MLA KV-cache behavior, but they
  do not route through PTO kernels.
- **Gap / next PTO milestone:** add PTO-owned decode, prefill, append,
  paged/ragged KV-cache, varlen, MLA, cascade, sparse, and POD attention
  fixtures before any attention-serving claim.
- **Explicit non-claim:** this is not FlashInfer integration evidence, not
  FlashInfer parity, and not simpler-nv/vLLM kernel integration evidence.

### GEMM And Linear Operations

- **FlashInfer reference family:** GEMM and linear operations: BF16 GEMM;
  FP8 GEMM; FP4 GEMM; Grouped GEMM.
- **PTO current evidence:** `gluon_gemm_h200.md` covers scalar FP32 GEMM
  correctness. `gluon_tensor_core_gemm.md` covers FP16-input tensor-core GEMM
  correctness on H200.
- **Gap / next PTO milestone:** add BF16, FP8, FP4, grouped GEMM, and
  linear-layer serving-shape fixtures with explicit dtype and shape
  boundaries.
- **Explicit non-claim:** this is not generated-kernel performance evidence
  and not proof of FlashInfer GEMM coverage.

### MoE

- **FlashInfer reference family:** MoE: Fused MoE Kernels; Multiple Routing
  Methods; Quantized MoE.
- **PTO current evidence:** `gluon_moe_expert_h200.md` covers one generated
  FP32 expert affine primitive. `persistent_moe_dispatch_combine_h200.md`
  covers a synthetic single-process persistent-device dispatch/combine graph.
- **Gap / next PTO milestone:** add fused MoE, routing-method, top-k routing,
  quantized expert, and distributed expert-parallel milestones before claiming
  serving MoE readiness.
- **Explicit non-claim:** this is not DeepSeek MoE serving evidence and not a
  claim that PTO implements FlashInfer fused MoE.

### Sampling And Decoding

- **FlashInfer reference family:** sampling and decoding: Sorting-Free
  Sampling; Top-K; Top-P; Min-P; Speculative Decoding.
- **PTO current evidence:** `gluon_topk_sampling_h200.md` records generated
  `topk_sampling_f32` Top-K correctness gates, preserving the default
  `rows=2, vocab=8, k=3` fixture and adding H200 evidence for
  `rows=3, vocab=16, k=5`. It validates deterministic CPU golden versus GPU
  result for both `values` and `indices`, checks result payload shapes before
  comparisons, and orders tied logits by lower token id first.
  `gluon_topp_sampling_h200.md` records one generated
  `topp_sampling_f32` Top-P correctness gate on H200, preserving the default
  `rows=2, vocab=8, max_k=5, p=0.75` fixture and adding H200 evidence for
  `rows=3, vocab=16, max_k=6, p=0.80`. It consumes probabilities that
  already sum to one, selects the smallest descending-probability prefix whose
  cumulative probability is at least `p`, fills unused output slots with
  `0.0` values and `-1` indices, validates `values`, `indices`,
  `selected_counts`, and `cumulative_probabilities`, and checks result
  payload shapes before comparisons.
  `gluon_minp_sampling_h200.md` records one generated `minp_sampling_f32`
  Min-P (min-p) correctness gate on H200 while preserving the default
  `rows=2, vocab=8, max_k=5, min_p=0.5` fixture and adding H200 evidence for
  `rows=3, vocab=16, max_k=6, min_p=0.5`.
  It consumes probabilities that already sum to one, selects tokens whose
  probability is at least `min_p * row_max_probability`, sorts by probability
  descending with lower token id first for ties, fills unused output slots
  with `0.0` values and `-1` indices, validates `values`, `indices`, and
  `selected_counts`, and checks result payload shapes before comparisons.
  `gluon_speculative_decoding_h200.md` records generated
  `speculative_accept_f32` Speculative Decoding accept/reject correctness
  gates, preserving the default `rows=2, max_draft=4` fixture and adding H200
  evidence for `rows=3, max_draft=6`. It consumes draft token ids, draft
  probabilities, target probabilities for those same draft tokens, and
  deterministic thresholds, accepts while
  `threshold <= min(1.0, target_probability / draft_probability)`, stops at
  first reject per row, fills later output ids with `-1`, and validates
  `accepted_token_ids`, `accept_mask`, and `accepted_counts`, checking result
  payload shapes before comparisons. vLLM probes use bounded sampler settings
  for real DeepSeek API requests, but those sampler settings do not route
  through PTO kernels.
  `pypto_serving_topk_sampling_launcher_h200.md` records a
  serving-route launcher/probe for the existing generated
  `topk_sampling_f32` Top-K correctness gate through the synthetic
  pypto-serving/simpler-nv source route. It preserves
  `launch_kind: gluon-topk-sampling`, shape
  `rows=3, vocab=16, k=5`, artifact/source digest metadata, and validation
  metadata.
- **Gap / next PTO milestone:** connect sampling to a serving stack. Remaining
  sampling gaps include serving-stack integration.
- **Explicit non-claim:** this is not FlashInfer integration evidence, not
  vLLM or simpler-nv kernel integration evidence, not tokenizer semantics,
  not generated-text correctness, and not DeepSeek serving through
  pypto-serving.

### Communication

- **FlashInfer reference family:** communication: AllReduce; Multi-Node
  NVLink; NVSHMEM Integration.
- **PTO current evidence:** `nccl_two_h200_baseline.md` records NCCL
  collectives on two H200 GPUs. UCCL adapter notes record same-node
  adapter/probe evidence, not a CUDA host-runtime UCCL ABI.
- **Gap / next PTO milestone:** keep NCCL as the baseline while adding
  serving-level communication milestones separately; multi-node NVLink and
  NVSHMEM need their own explicit gates.
- **Explicit non-claim:** this is not RDMA, multi-node, NVSHMEM, or
  serving-communication readiness evidence.

### Other Serving Operators

- **FlashInfer reference family:** other serving operators: RoPE;
  Normalization; Activations.
- **PTO current evidence:** CUDA vector and generated-kernel examples cover
  small arithmetic kernels. `gluon_rmsnorm_h200.md` records one generated
  `rmsnorm_f32` FP32 RMSNorm shape sweep on H200, including the existing
  smoke shape and `hidden=7168` with
  `DeepSeek-V4-Flash config hidden_size` provenance.
  `gluon_layernorm_h200.md` records one generated `layernorm_f32` FP32
  LayerNorm shape sweep on H200, including the existing smoke shape and
  `hidden=7168` with `DeepSeek-V4-Flash config hidden_size` provenance.
  `gluon_rope_h200.md` records one generated `rope_f32` FP32 RoPE shape
  sweep on H200, including the existing smoke shape and `head_dim=64` with
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  `rope_head_dim: 64` provenance. `gluon_silu_h200.md` records one generated
  `silu_f32` FP32 SiLU correctness sweep on H200, including the existing
  smoke shape and `moe_inter_dim: 2048` with `swiglu_limit: 10.0` provenance
  from
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  as standalone SiLU gate-activation-width evidence. `gluon_gelu_h200.md`
  records one generated `gelu_f32` FP32 GELU correctness sweep on H200,
  including the existing smoke shape and `moe_inter_dim: 2048` with
  `swiglu_limit: 10.0` provenance from
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`
  as standalone GELU activation-width evidence.
  `gluon_gated_silu_h200.md` records one generated `gated_silu_f32` FP32
  gated SiLU correctness sweep on H200, including the existing smoke shape
  and `moe_inter_dim: 2048` with `swiglu_limit: 10.0` provenance from
  `tmp/model-artifacts/deepseek-ai/DeepSeek-V4-Flash/inference/config.json`.
  `gluon_gemma_fused_rmsnorm_h200.md` records one generated
  `gemma_fused_rmsnorm_f32` FP32 Gemma-style fused norm correctness fixture on
  H200.
- **Gap / next PTO milestone:** add broader RoPE, normalization, and
  activation fixtures with model-shape provenance before treating the family
  as covered serving operators. Remaining gap: broader activation coverage
  and broader LayerNorm shape coverage beyond this sweep. Additional gap:
  additional non-RMSNorm normalization variants remain.
- **Explicit non-claim:** this is not production readiness evidence and not
  model semantic correctness. The RMSNorm, LayerNorm, and RoPE fixtures are
  not FlashInfer integration evidence, fused attention evidence, KV-cache
  integration evidence, activation coverage, or Gemma-style fused norm
  coverage. The SiLU sweep is not FlashInfer integration evidence, DeepSeek
  semantic correctness, GELU coverage, gated activation coverage, broader
  activation coverage, fused attention evidence, KV-cache integration
  evidence, throughput, latency, or vLLM/simpler-nv integration evidence. The
  GELU sweep is not FlashInfer integration evidence, production serving
  readiness, DeepSeek semantic correctness, SiLU coverage, gated activation
  coverage, broader activation coverage, fused attention evidence, KV-cache
  integration evidence, throughput, latency, or vLLM/simpler-nv integration
  evidence.
  The gated SiLU fixture is not FlashInfer integration evidence, fused
  attention evidence, KV-cache integration evidence, throughput, latency,
  DeepSeek semantic correctness, or vLLM/simpler-nv integration evidence.
  The Gemma-style fused norm fixture is not FlashInfer integration evidence,
  production serving readiness, DeepSeek semantic correctness, broader
  normalization coverage, activation coverage, fused attention evidence,
  KV-cache integration evidence, throughput, latency, or vLLM/simpler-nv
  integration evidence.

### H200 / Hopper Target

- **FlashInfer reference family:** H200 / Hopper target: Hopper SM 9.0
  includes H100 and H200.
- **PTO current evidence:** existing CUDA notes include H200 correctness or
  probe evidence for selected GEMM, attention, MoE, communication, and vLLM
  serving boundaries.
- **Gap / next PTO milestone:** for each serving operator family above,
  record an H200-specific pass/skip/fail gate and its non-claim boundary.
- **Explicit non-claim:** H200 availability does not imply FlashInfer
  integration, DeepSeek serving through pypto-serving, generated-kernel
  performance, or production readiness.

## Current Boundary

PTO now has a FlashInfer-derived operator checklist. That is a documentation
and review guard only.

Non-claims:

- This checklist is not FlashInfer integration evidence.
- This checklist is not DeepSeek serving through pypto-serving.
- This checklist is not generated-kernel performance evidence.
- This checklist is not production readiness evidence.
- This checklist is not simpler-nv/vLLM kernel integration evidence.

## Next Review Gate

Use this checklist when proposing PTO serving operator milestones. A future
operator slice should update one family-specific fixture or evidence note,
then update the matching gap without widening the non-claim boundary.
