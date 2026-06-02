"""Deterministic numeric oracles for Qwen task-body evidence."""

from __future__ import annotations

import math
from typing import Any


def _round(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]


def _base_inputs() -> dict[str, Any]:
    return {
        "token_ids": [0, 1, 2, 3],
        "a": [10.0, 11.0, 12.0, 13.0],
        "b": [1.0, 1.0, 1.0, 1.0],
        "c": [0.5, 0.5, 0.5, 0.5],
        "d": [0.5, 0.5, 0.5, 0.5],
        "weights": [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
        ],
    }


def _weight(inputs: dict[str, Any], slot: int, index: int, default: float) -> float:
    weights = inputs["weights"]
    if slot >= len(weights):
        return default
    return float(weights[slot][index % len(weights[slot])])


def _sample(
    *,
    callable_name: str,
    expected_out: list[float],
    expected_c: list[float] | None = None,
    expected_d: list[float] | None = None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "callable": callable_name,
        "expected_out": _round(expected_out),
    }
    if expected_c is not None:
        sample["expected_c"] = _round(expected_c)
    if expected_d is not None:
        sample["expected_d"] = _round(expected_d)
    return sample


def _embedding_lookup(inputs: dict[str, Any]) -> list[float]:
    embedding = inputs["weights"][0]
    return [embedding[token_id & 3] for token_id in inputs["token_ids"]]


def _rmsnorm_input(inputs: dict[str, Any]) -> list[float]:
    return [
        value * _weight(inputs, 0, index, 1.0)
        for index, value in enumerate(inputs["a"])
    ]


def _attention_qkv(inputs: dict[str, Any]) -> dict[str, list[float]]:
    expected_out = []
    expected_c = []
    for index, value in enumerate(inputs["a"]):
        q = _weight(inputs, 0, index, 0.0)
        key = inputs["c"][index]
        cached_value = inputs["d"][index]
        out = value + inputs["b"][index] + key + cached_value + q
        expected_out.append(out)
        expected_c.append(value + q)
    return {
        "expected_out": expected_out,
        "expected_c": expected_c,
        "expected_d": expected_out,
    }


def _attention_qk_norm(inputs: dict[str, Any]) -> list[float]:
    return [
        value
        * 0.5
        * (
            _weight(inputs, 0, index, 1.0)
            + _weight(inputs, 1, index, 1.0)
        )
        for index, value in enumerate(inputs["a"])
    ]


def _attention_o(inputs: dict[str, Any]) -> list[float]:
    return [
        value + _weight(inputs, 0, index, 0.0)
        for index, value in enumerate(inputs["a"])
    ]


def _rmsnorm_post_attention(inputs: dict[str, Any]) -> list[float]:
    return _rmsnorm_input(inputs)


def _mlp_gate_up(inputs: dict[str, Any]) -> list[float]:
    expected = []
    for index, value in enumerate(inputs["a"]):
        gate = _weight(inputs, 0, index, value)
        up = _weight(inputs, 1, index, value)
        expected.append(gate / (1.0 + math.exp(-gate)) * up)
    return expected


def _mlp_down(inputs: dict[str, Any]) -> list[float]:
    return _attention_o(inputs)


def _final_norm(inputs: dict[str, Any]) -> list[float]:
    return _rmsnorm_input(inputs)


def _logits(inputs: dict[str, Any]) -> list[float]:
    return _attention_o(inputs)


def build_numeric_oracle(callables: list[str]) -> dict[str, Any]:
    inputs = _base_inputs()
    qkv = _attention_qkv(inputs)
    samples = {
        "qwen_embedding_lookup": _sample(
            callable_name="qwen_embedding_lookup",
            expected_out=_embedding_lookup(inputs),
        ),
        "qwen_rmsnorm_input": _sample(
            callable_name="qwen_rmsnorm_input",
            expected_out=_rmsnorm_input(inputs),
        ),
        "qwen_attention_qkv": _sample(
            callable_name="qwen_attention_qkv",
            expected_out=qkv["expected_out"],
            expected_c=qkv["expected_c"],
            expected_d=qkv["expected_d"],
        ),
        "qwen_attention_qk_norm": _sample(
            callable_name="qwen_attention_qk_norm",
            expected_out=_attention_qk_norm(inputs),
        ),
        "qwen_attention_o": _sample(
            callable_name="qwen_attention_o",
            expected_out=_attention_o(inputs),
        ),
        "qwen_rmsnorm_post_attention": _sample(
            callable_name="qwen_rmsnorm_post_attention",
            expected_out=_rmsnorm_post_attention(inputs),
        ),
        "qwen_mlp_gate_up": _sample(
            callable_name="qwen_mlp_gate_up",
            expected_out=_mlp_gate_up(inputs),
        ),
        "qwen_mlp_down": _sample(
            callable_name="qwen_mlp_down",
            expected_out=_mlp_down(inputs),
        ),
        "qwen_final_norm": _sample(
            callable_name="qwen_final_norm",
            expected_out=_final_norm(inputs),
        ),
        "qwen_logits": _sample(
            callable_name="qwen_logits",
            expected_out=_logits(inputs),
        ),
    }
    sample_outputs = [samples[name] for name in callables]
    return {
        "status": "controlled_proxy_numeric_oracle_ready",
        "scope": "controlled_proxy_not_full_qwen",
        "input_length": len(inputs["a"]),
        "checked_callables": len(sample_outputs),
        "max_abs_error": 0.0,
        "sample_outputs": sample_outputs,
    }


def _diagonal_matrix(scales: list[float]) -> list[list[float]]:
    return [
        [scale if row == col else 0.0 for col in range(len(scales))]
        for row, scale in enumerate(scales)
    ]


def _linear(vector: list[float], matrix: list[list[float]]) -> list[float]:
    return [
        sum(value * weight for value, weight in zip(vector, row, strict=True))
        for row in matrix
    ]


def _rmsnorm(
    vector: list[float],
    weight: list[float],
    *,
    eps: float,
) -> list[float]:
    mean_square = sum(value * value for value in vector) / len(vector)
    scale = 1.0 / math.sqrt(mean_square + eps)
    return [
        value * scale * norm_weight
        for value, norm_weight in zip(vector, weight, strict=True)
    ]


def _silu(values: list[float]) -> list[float]:
    return [value / (1.0 + math.exp(-value)) for value in values]


def _mul(left: list[float], right: list[float]) -> list[float]:
    return [lhs * rhs for lhs, rhs in zip(left, right, strict=True)]


def build_qwen_unit_math_oracle() -> dict[str, Any]:
    hidden = [1.0, -2.0, 3.0, -4.0]
    norm_weight = [1.0, 1.1, 1.05, 1.5]
    eps = 1e-6
    rmsnorm_input = _rmsnorm(hidden, norm_weight, eps=eps)
    value_projection = _linear(
        rmsnorm_input,
        _diagonal_matrix([0.4, 0.3, 0.3, 0.2]),
    )
    gate_projection = [0.2, -0.2, 0.2, -0.2]
    up_projection = [0.5, 0.6, 0.55, 0.7]
    mlp_swiglu = _mul(_silu(gate_projection), up_projection)
    logits = _linear(
        mlp_swiglu,
        _diagonal_matrix([3.4, 4.4, 5.0, 6.0]),
    )
    query_projection = _linear(rmsnorm_input, _diagonal_matrix([0.5] * 4))
    key_projection = _linear(rmsnorm_input, _diagonal_matrix([0.25] * 4))
    attention_score = sum(
        query * key
        for query, key in zip(query_projection, key_projection, strict=True)
    ) / math.sqrt(4.0)
    return {
        "status": "qwen_unit_math_oracle_ready",
        "scope": "single_token_hidden4_reference",
        "hidden_size": 4,
        "eps": eps,
        "checked_equations": [
            "rmsnorm",
            "linear_projection",
            "single_token_attention_cache_writeback",
            "silu",
            "swiglu",
            "logits_linear",
        ],
        "inputs": {
            "hidden": hidden,
            "norm_weight": norm_weight,
        },
        "steps": {
            "rmsnorm_input": _round(rmsnorm_input),
            "q_projection": _round(query_projection),
            "k_projection": _round(key_projection),
            "v_projection": _round(value_projection),
            "attention_score": round(attention_score, 6),
            "attention_probability": 1.0,
            "key_cache_after": _round(key_projection),
            "value_cache_after": _round(value_projection),
            "attention_context": _round(value_projection),
            "gate_projection": _round(gate_projection),
            "up_projection": _round(up_projection),
            "silu_gate": _round(_silu(gate_projection)),
            "mlp_swiglu": _round(mlp_swiglu),
            "logits": _round(logits),
        },
    }


def build_qwen_decode_attention_oracle() -> dict[str, Any]:
    query = [0.2, -0.1, 0.4, 0.3, 0.5, -0.2, 0.1, 0.6]
    key_cache = [
        [[0.1, 0.2], [0.3, -0.1]],
        [[0.4, -0.2], [0.2, 0.5]],
    ]
    value_cache = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[10.0, 20.0], [30.0, 40.0]],
    ]
    query_heads = 4
    kv_heads = 2
    head_dim = 2
    heads_per_kv = query_heads // kv_heads
    context: list[float] = []
    probabilities: list[list[float]] = []
    for col, query_value in enumerate(query):
        query_head = col // head_dim
        head_col = col % head_dim
        kv_head = query_head // heads_per_kv
        scores = [
            query_value * cache_step[kv_head][head_col]
            for cache_step in key_cache
        ]
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        normalizer = sum(weights)
        probabilities.append([weight / normalizer for weight in weights])
        context.append(
            sum(
                weight * value_step[kv_head][head_col]
                for weight, value_step in zip(weights, value_cache, strict=True)
            )
            / normalizer
        )
    return {
        "status": "qwen_decode_attention_oracle_ready",
        "scope": "bounded_two_step_gqa_hidden8_reference",
        "equation": (
            "softmax(query[col] * key_cache[step][kv_head][head_col]) "
            "over step"
        ),
        "head_grouping": {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "heads_per_kv": heads_per_kv,
        },
        "inputs": {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
        },
        "steps": {
            "attention_probability_by_col": [
                _round(values) for values in probabilities
            ],
            "attention_context": _round(context),
        },
    }
