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
  JSON and remains separate from the promoted sweep. The same note now records
  bounded causal prefill sweep correctness evidence for
  `--sweep --causal --require-cuda`, with `schema_version`, aggregate status,
  `case_count`, only
  causal prefill cases, same-length shapes through `prefill_64x64x64`,
  `phase: prefill`, `causal: true`, tolerance, provenance, the
  lower-triangular masked PyTorch reference formula, status, and repo-relative
  artifact paths. It also records
  bounded causal decode sweep correctness evidence for
  `--sweep --causal --causal-sweep-phase decode --require-cuda`, with
  `schema_version`, aggregate status, `case_count`, only causal decode cases,
  decode shapes, `phase: decode`, `causal: true`, tolerance, provenance, the
  shifted masked PyTorch reference formula, status, and repo-relative artifact
  paths. The promoted decode sweep includes `decode_1x64x64` and
  `decode_1x128x32`; `1x128x64 hit a Triton CUDA out-of-memory boundary`, so
  it is not claimed as passing. This is bounded causal decode sweep
  correctness evidence. It also records bounded causal append sweep
  correctness evidence for
  `--sweep --causal --causal-sweep-phase append --require-cuda`, with
  `schema_version`, aggregate status, `case_count`, only causal append cases,
  append shapes, `phase: append`, `causal: true`, tolerance, provenance, the
  shifted masked PyTorch reference formula, status, and repo-relative artifact
  paths. This is bounded causal append sweep correctness evidence. The same
  note also records
  a bounded same-length multi-query prefill-shaped H200 gate for
  `--tile-shape 32x32x64 --causal` with `phase: prefill`, `causal: true`,
  shape, tolerance, the lower-triangular masked PyTorch reference formula,
  status, and repo-relative artifact paths. It also records a bounded
  single-query decode-shaped H200 gate for `--tile-shape 1x32x64 --causal`
  with `phase: decode`, `causal: true`, shape, tolerance, the offset masked
  PyTorch reference formula, status, and repo-relative artifact paths. It also
  records a small multi-query append-shaped H200 gate for
  `--tile-shape 4x32x64 --causal` with `phase: append`, `causal: true`,
  shape, tolerance, the offset masked PyTorch reference formula, status, and
  repo-relative artifact paths. It also records explicit paged/ragged
  KV-cache unsupported-boundary evidence for
  `--tile-shape 32x32x64 --causal --kv-cache-boundary paged` and
  `--tile-shape 32x32x64 --causal --kv-cache-boundary ragged`, with
  `status: skipped`, `unsupported_boundary.kind: paged_kv_cache`,
  `unsupported_boundary.kind: ragged_kv_cache`, shape metadata, and
  repo-relative, private-path-safe commands. It also records explicit varlen
  unsupported-boundary evidence for
  `--tile-shape 32x32x64 --causal --sequence-boundary varlen`, with
  `status: skipped`, `sequence_boundary: varlen`,
  `unsupported_boundary.kind: varlen_attention`, shape metadata, and
  repo-relative, private-path-safe commands. It also records explicit MLA
  unsupported-boundary evidence for
  `--tile-shape 32x32x64 --causal --attention-variant mla`, with
  `status: skipped`, `attention_variant: mla`,
  `unsupported_boundary.kind: mla_attention`, shape metadata, and
  repo-relative, private-path-safe commands. It also records explicit
  Cascade Attention unsupported-boundary evidence for
  `--tile-shape 32x32x64 --causal --attention-variant cascade`, with
  `status: skipped`, `attention_variant: cascade`,
  `unsupported_boundary.kind: cascade_attention`, shape metadata, and
  repo-relative, private-path-safe commands. This is unsupported-boundary
  evidence only. It also records explicit Sparse Attention
  unsupported-boundary evidence for
  `--tile-shape 32x32x64 --causal --attention-variant sparse`, with
  `status: skipped`, `attention_variant: sparse`,
  `unsupported_boundary.kind: sparse_attention`, shape metadata, and
  repo-relative, private-path-safe commands. This is unsupported-boundary
  evidence only. It also records explicit POD-Attention unsupported-boundary
  evidence for `--tile-shape 32x32x64 --causal --attention-variant pod`, with
  `status: skipped`, `attention_variant: pod`,
  `unsupported_boundary.kind: pod_attention`, shape metadata, and
  repo-relative, private-path-safe commands. This is unsupported-boundary
  evidence only. vLLM DeepSeek probes are real vLLM serving evidence and
  mention fp8 MLA KV-cache behavior, but they do not route through PTO
  kernels. This is not Cascade Attention correctness. This is not Sparse
  Attention correctness. This is not POD-Attention correctness.
- **Gap / next PTO milestone:** add PTO-owned full prefill, append,
  paged/ragged KV-cache correctness, varlen, full decode coverage, MLA,
  cascade, sparse, and POD-Attention correctness fixtures before any
  attention-serving
  claim.
- **Explicit non-claim:** this is not MLA attention correctness, not
  Cascade Attention correctness, not Sparse Attention correctness, not
  POD-Attention correctness, not FlashInfer integration evidence, not
  FlashInfer parity, not
  simpler-nv/vLLM kernel integration evidence, not production serving
  readiness, not performance/throughput/latency evidence,
  not paged/ragged KV-cache correctness, not full prefill coverage, not full
  decode, not full append, not attention-variant correctness, not full
  prefill, full decode, full append, or append KV-cache coverage,
  not bounded append KV-cache coverage,
  not varlen attention correctness, and not DeepSeek semantic correctness.

### GEMM And Linear Operations

- **FlashInfer reference family:** GEMM and linear operations: BF16 GEMM;
  FP8 GEMM; FP4 GEMM; Grouped GEMM.
- **PTO current evidence:** `gluon_gemm_h200.md` covers scalar FP32 GEMM
  correctness. `gluon_tensor_core_gemm.md` covers FP16-input tensor-core GEMM
  correctness on H200. It also records BF16 tensor-core GEMM correctness on
  H200 for `gemm_tensor_core_tiled_bf16_f32` with BF16 inputs and FP32
  accumulator/output. The BF16 sweep is guarded by
  `examples/cuda/gluon_wgmma_api_preflight.py`; the fresh project-local
  `.venv` preflight failed because Torch and Triton were absent, while the
  preserved Gluon environment preflight passed before the BF16 run. The BF16
  sweep covered a smoke tile and a bounded `m=64,k=7168,n=128` linear-style
  shape using `DeepSeek-V4-Flash config hidden_size=7168` provenance; case
  statuses: passed, passed; largest max absolute error:
  `0.002899169921875`. The FP8 boundary harness records that
  `torch.float8_e4m3fn` and Gluon `gl.float8e4nv` are visible in the
  preserved H200 Gluon environment, then fails at WGMMA lowering with
  `PassManager::run failed` and the compiler assertion
  `WGMMA type or shape is not supported`. This is unsupported-boundary
  evidence only, not FP8 GEMM correctness evidence. The FP4 boundary harness
  records that `torch.float4_e2m1fn_x2` is visible, but the only Gluon
  FP4-related attr found is `fp4_to_fp`; no Gluon FP4 WGMMA dtype is
  available. The H200 FP4 gate exits as `status: skipped` with `artifact:
  null`, `unsupported_boundary.kind: gluon_fp4_dtype_api_unavailable`, and
  reason `missing Gluon FP4 WGMMA dtype API`. This is unsupported-boundary
  evidence only, not FP4 GEMM correctness evidence. The Grouped GEMM boundary
  harness records proposed grouped shapes, probes the generated-kernel
  registry plus actual grouped GEMM Gluon/Hopper attrs, and records generic
  Hopper `warpgroup_mma*` primitives separately from grouped GEMM API/source
  support. It exits as `status: skipped` with
  `artifact: null`,
  `unsupported_boundary.kind: gluon_grouped_gemm_source_path_unavailable`,
  and reason `missing grouped GEMM WGMMA source path`. This is
  unsupported-boundary evidence only, not grouped GEMM correctness evidence.
- **Gap / next PTO milestone:** make FP8 WGMMA lowering pass before promoting
  FP8 GEMM correctness evidence; make a Gluon FP4 WGMMA dtype/lowering path
  available before promoting FP4 GEMM correctness evidence. The FP4
  API/lowering boundary is explicitly recorded. Add a grouped GEMM source,
  lowering, and runtime correctness path before promoting grouped GEMM
  correctness evidence. Add broader linear-layer serving-shape fixtures with
  explicit dtype and shape boundaries before any FlashInfer GEMM coverage
  claim.
- **Explicit non-claim:** this is not generated-kernel performance evidence
  and not proof of FlashInfer GEMM coverage. The BF16 tensor-core GEMM
  correctness evidence is not FlashInfer integration evidence, not
  vLLM/simpler-nv serving integration, and not production readiness evidence.
  The FP8 boundary is not FlashInfer integration evidence, not serving
  integration evidence, not generated-kernel performance evidence, not
  production readiness evidence, and not BF16/FP4/grouped
  GEMM/MoE/FlashAttention/vLLM integration evidence. The FP4 boundary is not
  FlashInfer integration evidence, not serving integration evidence, not
  generated-kernel performance evidence, not production readiness evidence,
  and not BF16/FP8/grouped GEMM/MoE/FlashAttention/vLLM integration evidence.
  The grouped GEMM boundary is not FlashInfer integration evidence, not serving
  integration evidence, not generated-kernel performance evidence, not
  production readiness evidence, and not BF16/FP8/FP4
  GEMM/MoE/FlashAttention/vLLM integration evidence.

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
  `pypto_serving_topp_sampling_launcher_h200.md` records a
  serving-route launcher/probe for the existing generated
  `topp_sampling_f32` Top-P correctness gate through the synthetic
  pypto-serving/simpler-nv source route. It preserves
  `launch_kind: gluon-topp-sampling`, shape
  `rows=3, vocab=16, max_k=6, p=0.80`, artifact/source digest metadata, and
  validation metadata.
  `pypto_serving_minp_sampling_launcher_h200.md` records a
  serving-route launcher/probe for the existing generated
  `minp_sampling_f32` Min-P correctness gate through the synthetic
  pypto-serving/simpler-nv source route. It preserves
  `launch_kind: gluon-minp-sampling`, shape
  `rows=3, vocab=16, max_k=6, min_p=0.5`, artifact/source digest metadata,
  and validation metadata.
  `pypto_serving_speculative_decoding_launcher_h200.md` records a
  serving-route launcher/probe for the existing generated
  `speculative_accept_f32` speculative decoding accept/reject correctness
  gate through the synthetic pypto-serving/simpler-nv source route. It
  preserves `launch_kind: gluon-speculative-decoding`, shape
  `rows=3, max_draft=6`, artifact/source digest metadata, and accepted-token,
  accept-mask, and accepted-count validation metadata.
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
