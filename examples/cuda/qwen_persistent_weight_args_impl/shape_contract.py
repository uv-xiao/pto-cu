"""Qwen task-shape fields for CUDA persistent weight descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QwenTaskShape:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    vocab_size: int = 151936
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128

    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim


QWEN3_8B_TASK_SHAPE = QwenTaskShape()
KV_PAGE_SIZE_TOKENS = 16
LOGITS_TILE_SIZE = 256
LOGITS_DIAGNOSTIC_ACTIVE_COLUMNS = 1024
PROJECTION_DIAGNOSTIC_ACTIVE_COLUMNS = 1024


def shape_contract_payload(shape: QwenTaskShape) -> dict[str, Any]:
    return {
        "model_family": "Qwen3",
        "hidden_size": shape.hidden_size,
        "intermediate_size": shape.intermediate_size,
        "vocab_size": shape.vocab_size,
        "num_attention_heads": shape.num_attention_heads,
        "num_key_value_heads": shape.num_key_value_heads,
        "head_dim": shape.head_dim,
        "q_width": shape.q_width,
        "kv_width": shape.kv_width,
        "field_policy": (
            "workload rows come from decode args; descriptor cols/inner/leading "
            "dimensions describe callable-local matrix shape"
        ),
    }


def task_shape_fields(callable_name: str, shape: QwenTaskShape) -> dict[str, Any]:
    if callable_name == "qwen_attention_qkv":
        cols = shape.q_width + 2 * shape.kv_width
        fields = matrix_fields(cols=cols, inner=shape.hidden_size)
        fields["scalar0"] = KV_PAGE_SIZE_TOKENS
        fields["scalar1"] = PROJECTION_DIAGNOSTIC_ACTIVE_COLUMNS
        return fields
    if callable_name == "qwen_attention_o":
        return attention_fields(
            cols=shape.hidden_size,
            query_heads=shape.num_attention_heads,
            kv_heads=shape.num_key_value_heads,
            head_dim=shape.head_dim,
        )
    if callable_name == "qwen_mlp_gate_up":
        fields = matrix_fields(cols=shape.intermediate_size, inner=shape.hidden_size)
        fields["scalar1"] = PROJECTION_DIAGNOSTIC_ACTIVE_COLUMNS
        return fields
    if callable_name == "qwen_mlp_down":
        fields = matrix_fields(cols=shape.hidden_size, inner=shape.intermediate_size)
        fields["scalar1"] = PROJECTION_DIAGNOSTIC_ACTIVE_COLUMNS
        return fields
    if callable_name == "qwen_logits":
        fields = matrix_fields(cols=shape.vocab_size, inner=shape.hidden_size)
        fields["scalar0"] = LOGITS_TILE_SIZE
        fields["scalar1"] = LOGITS_DIAGNOSTIC_ACTIVE_COLUMNS
        return fields
    if callable_name in {
        "qwen_rmsnorm_input",
        "qwen_rmsnorm_post_attention",
        "qwen_final_norm",
    }:
        return vector_fields(width=shape.hidden_size)
    if callable_name == "qwen_attention_qk_norm":
        return {
            "rows": shape.num_attention_heads,
            "cols": shape.q_width + shape.kv_width,
            "inner": shape.head_dim,
            "lda": shape.head_dim,
            "ldb": shape.num_key_value_heads,
            "ldc": shape.q_width + shape.kv_width,
            "scalar0": KV_PAGE_SIZE_TOKENS,
        }
    if callable_name == "qwen_embedding_lookup":
        return {
            "cols": shape.hidden_size,
            "inner": shape.hidden_size,
            "ldb": shape.hidden_size,
        }
    return {}


def matrix_fields(*, cols: int, inner: int) -> dict[str, int]:
    return {
        "cols": cols,
        "inner": inner,
        "lda": inner,
        "ldb": inner,
        "ldc": cols,
    }


def attention_fields(
    *,
    cols: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    qk_norm_stride = (query_heads + kv_heads) * head_dim
    return {
        "rows": query_heads,
        "cols": cols,
        "inner": head_dim,
        "lda": head_dim,
        "ldb": kv_heads,
        "ldc": cols,
        "a_batch_stride": qk_norm_stride,
        "scalar0": 16.0,
        "scalar1": 16.0,
    }


def vector_fields(*, width: int) -> dict[str, int]:
    return {
        "cols": width,
        "inner": width,
        "lda": width,
        "ldb": width,
        "ldc": width,
    }
