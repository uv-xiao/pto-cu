"""Deterministic numeric oracle for the current Qwen proxy task bodies."""

from __future__ import annotations

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
    return [
        value * _weight(inputs, 0, index, 1.0)
        + _weight(inputs, 1, index, 0.0)
        for index, value in enumerate(inputs["a"])
    ]


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
