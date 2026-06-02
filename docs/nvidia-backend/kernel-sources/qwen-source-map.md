# Qwen Kernel Source Map

This source map records external CUDA implementations used only as design
references for the PTO Qwen task bodies. Local sparse snapshots are kept under
`tmp/sources/kernel-references/` for review; no upstream repositories are
modified and no external kernel source is vendored into PTO.

## Reference Snapshots

| Project | Commit | Local Snapshot |
| --- | --- | --- |
| FlashInfer | `c3c40a7` | `tmp/sources/kernel-references/flashinfer` |
| vLLM | `1edfd09` | `tmp/sources/kernel-references/vllm` |
| SGLang | `1033d83` | `tmp/sources/kernel-references/sglang` |

## PTO Mapping

| PTO Callable Area | Reference Kernel Families | Current PTO Status |
| --- | --- | --- |
| RMSNorm callables | FlashInfer `norm.cu`, vLLM `layernorm_kernels.cu`, SGLang `rmsnorm.cuh` | Block-threaded diagnostic reduction is live. |
| MLP gate/up | vLLM `activation_kernels.cu`, SGLang SiLU-mul JIT kernels | Elementwise unit-math formula is live; model-shape gate/up buffers remain missing. |
| QK norm, RoPE, attention | FlashInfer decode/QK-RoPE, SGLang fused QKNorm-RoPE | PTO has shape-field QK RMSNorm/RoPE formula source, descriptor-level RoPE table slots, and launch-packet live RoPE pointer binding with Qwen-theta first-position table population; dynamic per-step RoPE refresh and decode attention remain open. |
| KV cache writes | vLLM `cache_kernels.cu`, SGLang `kvcache.cuh` | PTO has mutable `c`/`d` fields; slot-mapped cache writes remain missing. |
| Logits | vLLM Qwen3 model path | PTO has diagnostic logits and sampled-token feedback; tiled vocab projection remains missing. |

The machine-readable copy is emitted in the Qwen task-body manifest as
`qwen_kernel_source_map`.
