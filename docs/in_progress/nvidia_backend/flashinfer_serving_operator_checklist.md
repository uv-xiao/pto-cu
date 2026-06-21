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
- **PTO current evidence:** `gluon_flashattention_h200.md` records a
  single-tile FP32 FlashAttention forward correctness case. vLLM DeepSeek
  probes are real vLLM serving evidence and mention fp8 MLA KV-cache behavior,
  but they do not route through PTO kernels.
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
- **PTO current evidence:** vLLM probes use bounded sampler settings for real
  DeepSeek API requests. PTO has no checked-in sampling kernel or speculative
  decode operator evidence.
- **Gap / next PTO milestone:** add PTO sampling-kernel fixtures for top-k,
  top-p, min-p, and a separate speculative decoding boundary before connecting
  sampling to a serving stack.
- **Explicit non-claim:** this is not tokenizer semantics, generated-text
  correctness, or DeepSeek serving through pypto-serving.

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
  `rmsnorm_f32` FP32 RMSNorm correctness fixture on H200.
  `gluon_layernorm_h200.md` records one generated `layernorm_f32` FP32
  LayerNorm correctness fixture on H200.
  `gluon_rope_h200.md` records one generated `rope_f32` FP32 RoPE
  correctness fixture on H200. `gluon_silu_h200.md` records one generated
  `silu_f32` FP32 SiLU correctness fixture on H200. `gluon_gelu_h200.md`
  records one generated `gelu_f32` FP32 GELU correctness fixture on H200.
  `gluon_gated_silu_h200.md` records one generated `gated_silu_f32` FP32
  gated SiLU correctness fixture on H200. Remaining normalization gaps include
  Gemma-style fused norm.
- **Gap / next PTO milestone:** add broader RoPE, normalization, and
  activation fixtures with model-shape provenance before treating the family
  as covered serving operators. The next normalization gaps are broader
  RMSNorm and LayerNorm shape coverage plus Gemma-style fused norm.
- **Explicit non-claim:** this is not production readiness evidence and not
  model semantic correctness. The RMSNorm, LayerNorm, and RoPE fixtures are
  not FlashInfer integration evidence, fused attention evidence, KV-cache
  integration evidence, activation coverage, or Gemma-style fused norm
  coverage. The SiLU fixture is not FlashInfer integration evidence, GELU
  coverage, gated activation coverage, fused attention evidence, KV-cache
  integration evidence, throughput, latency, or vLLM/simpler-nv integration
  evidence. The GELU fixture is not FlashInfer integration evidence, gated
  activation coverage, fused attention evidence, KV-cache integration
  evidence, throughput, latency, or vLLM/simpler-nv integration evidence.
  The gated SiLU fixture is not FlashInfer integration evidence, fused
  attention evidence, KV-cache integration evidence, throughput, latency,
  DeepSeek semantic correctness, or vLLM/simpler-nv integration evidence.

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
