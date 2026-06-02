"""External kernel source references for Qwen task-body hardening."""

from __future__ import annotations

from typing import Any


REFERENCE_SNAPSHOTS = [
    {
        "project": "FlashInfer",
        "commit": "c3c40a7",
        "local_snapshot": "tmp/sources/kernel-references/flashinfer",
        "upstream": "https://github.com/flashinfer-ai/flashinfer",
    },
    {
        "project": "vLLM",
        "commit": "1edfd09",
        "local_snapshot": "tmp/sources/kernel-references/vllm",
        "upstream": "https://github.com/vllm-project/vllm",
    },
    {
        "project": "SGLang",
        "commit": "1033d83",
        "local_snapshot": "tmp/sources/kernel-references/sglang",
        "upstream": "https://github.com/sgl-project/sglang",
    },
]


KERNEL_SOURCE_MAP = [
    {
        "pto_callables": [
            "qwen_rmsnorm_input",
            "qwen_rmsnorm_post_attention",
            "qwen_final_norm",
        ],
        "required_semantics": "RMSNorm over full hidden vectors with weight scale.",
        "reference_files": [
            {
                "project": "vLLM",
                "path": "csrc/libtorch_stable/layernorm_kernels.cu",
                "symbols": ["rms_norm_kernel", "fused_add_rms_norm_kernel"],
                "url": "https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu",
            },
            {
                "project": "FlashInfer",
                "path": "csrc/norm.cu",
                "symbols": ["rmsnorm", "fused_add_rmsnorm"],
                "url": "https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/norm.cu",
            },
            {
                "project": "SGLang",
                "path": "python/sglang/jit_kernel/csrc/elementwise/rmsnorm.cuh",
                "symbols": ["rmsnorm_cta"],
                "url": "https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/elementwise/rmsnorm.cuh",
            },
        ],
        "pto_status": "block_threaded_diagnostic_ready",
        "next_step": "Preserve the current block reduction but switch from diagnostic float-only formulas to model-shape dtype-aware row contracts.",
    },
    {
        "pto_callables": ["qwen_mlp_gate_up"],
        "required_semantics": "SiLU gate times up projection output for SwiGLU.",
        "reference_files": [
            {
                "project": "vLLM",
                "path": "csrc/libtorch_stable/activation_kernels.cu",
                "symbols": ["act_and_mul_kernel", "silu_and_mul"],
                "url": "https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/activation_kernels.cu",
            },
            {
                "project": "SGLang",
                "path": "python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh",
                "symbols": ["silu_and_mul"],
                "url": "https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh",
            },
        ],
        "pto_status": "elementwise_unit_math_ready",
        "next_step": "Make gate/up projection outputs explicit buffers, then run the same SiLU-mul formula over model-shape MLP intermediate rows.",
    },
    {
        "pto_callables": [
            "qwen_attention_qkv",
            "qwen_attention_qk_norm",
            "qwen_attention_o",
        ],
        "required_semantics": "Q/K/V projection, QK RMSNorm, RoPE, KV-cache writeback, and decode attention.",
        "reference_files": [
            {
                "project": "FlashInfer",
                "path": "csrc/norm.cu",
                "symbols": ["fused_qk_rmsnorm_rope_run"],
                "url": "https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/norm.cu",
            },
            {
                "project": "FlashInfer",
                "path": "csrc/batch_decode.cu",
                "symbols": ["BatchDecodeWithPagedKVCache"],
                "url": "https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/batch_decode.cu",
            },
            {
                "project": "SGLang",
                "path": "python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh",
                "symbols": ["fusedQKNormRopeKernel"],
                "url": "https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh",
            },
        ],
        "pto_status": "qkv_projection_qk_rmsnorm_rope_and_bounded_attention_source_ready",
        "next_step": "Replace the bounded per-column diagnostic attention reduction with model-shape head grouping, paged KV-cache addressing, and tiled softmax before full-serving promotion.",
    },
    {
        "pto_callables": ["qwen_attention_qkv"],
        "required_semantics": "Coalesced key/value writes into persistent KV-cache slots.",
        "reference_files": [
            {
                "project": "vLLM",
                "path": "csrc/libtorch_stable/cache_kernels.cu",
                "symbols": ["reshape_and_cache_kernel"],
                "url": "https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu",
            },
            {
                "project": "SGLang",
                "path": "python/sglang/jit_kernel/csrc/elementwise/kvcache.cuh",
                "symbols": ["store_kvcache"],
                "url": "https://github.com/sgl-project/sglang/blob/main/python/sglang/jit_kernel/csrc/elementwise/kvcache.cuh",
            },
        ],
        "pto_status": "mutable_c_d_fields_ready",
        "next_step": "Replace flat diagnostic c/d writes with slot-mapped KV-cache row writes matching the serving decode plan.",
    },
    {
        "pto_callables": ["qwen_logits"],
        "required_semantics": "Final hidden-to-vocab projection plus sampled-token feedback.",
        "reference_files": [
            {
                "project": "vLLM",
                "path": "vllm/model_executor/models/qwen3.py",
                "symbols": ["Qwen3ForCausalLM", "ParallelLMHead"],
                "url": "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py",
            },
        ],
        "pto_status": "diagnostic_logits_formula_ready",
        "next_step": "Lower the logits task to tiled hidden-by-vocab projection before treating sampled-token feedback as model-correct.",
    },
]


def build_kernel_source_map() -> dict[str, Any]:
    return {
        "status": "qwen_kernel_source_map_ready",
        "source_note": "tmp/sources/kernel-references/qwen-kernel-source-map.md",
        "reference_snapshots": REFERENCE_SNAPSHOTS,
        "entries": KERNEL_SOURCE_MAP,
    }
