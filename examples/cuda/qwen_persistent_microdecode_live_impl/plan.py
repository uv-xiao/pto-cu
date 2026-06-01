"""Reviewable plan for the controlled Qwen microdecode live proxy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_persistent_proxy_live_impl.plan import repo_relative, write_json
from qwen_persistent_task_bodies_impl.lifecycle import FUNC_ID_BASE


CALLABLES = [
    ("qwen_attention_qkv", FUNC_ID_BASE + 2),
    ("qwen_attention_o", FUNC_ID_BASE + 4),
    ("qwen_logits", FUNC_ID_BASE + 9),
]


def _round(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]


def _inputs() -> dict[str, Any]:
    return {
        "a": [10.0, 11.0, 12.0, 13.0],
        "b": [1.0, 1.0, 1.0, 1.0],
        "c": [0.5, 0.5, 0.5, 0.5],
        "d": [0.5, 0.5, 0.5, 0.5],
        "weights": [
            [1.0, 2.0, 3.0, 4.0],
            [0.5, 1.5, 2.5, 3.5],
        ],
    }


def _expected_for_step(inputs: dict[str, Any], c_values: list[float], d_values: list[float]) -> dict[str, list[float]]:
    qkv_out = []
    qkv_c = []
    for index, value in enumerate(inputs["a"]):
        q = inputs["weights"][0][index]
        out = value + inputs["b"][index] + c_values[index] + d_values[index] + q
        qkv_out.append(out)
        qkv_c.append(value + q)
    attention_o = [
        value + inputs["weights"][1][index]
        for index, value in enumerate(qkv_out)
    ]
    logits = [
        value + inputs["weights"][0][index]
        for index, value in enumerate(attention_o)
    ]
    return {
        "attention_qkv_out": _round(qkv_out),
        "attention_o_out": _round(attention_o),
        "logits_out": _round(logits),
        "c": _round(qkv_c),
        "d": _round(qkv_out),
    }


def _decode_iterations(inputs: dict[str, Any], repeat_runs: int) -> list[dict[str, Any]]:
    c_values = list(inputs["c"])
    d_values = list(inputs["d"])
    iterations = []
    for index in range(repeat_runs):
        expected = _expected_for_step(inputs, c_values, d_values)
        iterations.append({"iteration": index, "expected": expected})
        c_values = expected["c"]
        d_values = expected["d"]
    return iterations


def build_live_microdecode_plan(
    *,
    scheduler_blocks: int = 1,
    worker_blocks: int = 1,
    queue_capacity: int = 8,
    block_dim: int = 128,
    repeat_runs: int = 1,
) -> dict[str, Any]:
    if repeat_runs <= 0:
        raise ValueError("repeat_runs must be positive")
    inputs = _inputs()
    decode_iterations = _decode_iterations(inputs, repeat_runs)
    return {
        "kind": "pto_qwen_microdecode_live_execution_plan",
        "status": "ready_to_run",
        "scope": "controlled_proxy_not_full_qwen",
        "model_id": "Qwen/Qwen3-8B",
        "runtime": "cuda/persistent_device",
        "dag": {
            "task_count": 3,
            "dependent_count": 2,
            "scheduler_blocks": scheduler_blocks,
            "worker_blocks": worker_blocks,
            "queue_capacity": queue_capacity,
            "block_dim": block_dim,
            "dependency_edges": [
                ["qwen_attention_qkv", "qwen_attention_o"],
                ["qwen_attention_o", "qwen_logits"],
            ],
        },
        "decode_loop": {
            "repeat_runs": repeat_runs,
            "planned_task_executions": repeat_runs * 3,
            "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
            "reset_between_runs": [
                "fanin",
                "ready_flags",
                "completion_flags",
                "counters",
            ],
            "carried_between_runs": [
                "key_cache_mutable",
                "value_cache_mutable",
            ],
        },
        "tasks": [
            {
                "callable": name,
                "func_id": func_id,
                "initial_fanin": 0 if index == 0 else 1,
            }
            for index, (name, func_id) in enumerate(CALLABLES)
        ],
        "task_argument_fields": {
            "qwen_attention_qkv": {
                "a": "hidden_state",
                "b": "attention_mask",
                "out": "attention_qkv_out",
                "c": "key_cache_mutable",
                "d": "value_cache_mutable",
                "tensor_args[0]": "q_proj_weight",
            },
            "qwen_attention_o": {
                "a": "attention_qkv_out",
                "out": "attention_o_out",
                "tensor_args[0]": "o_proj_weight",
            },
            "qwen_logits": {
                "a": "attention_o_out",
                "out": "logits_out",
                "tensor_args[0]": "lm_head_weight",
            },
        },
        "inputs": inputs,
        "decode_iterations": decode_iterations,
        "expected": decode_iterations[-1]["expected"],
        "implemented_contracts": [
            "controlled_proxy_live_microdecode_plan",
            "controlled_proxy_live_decode_loop_plan",
            "persistent_device_proxy_decode_chain_plan",
            "qwen_attention_to_logits_dag_contract",
        ],
        "remaining_runtime_gaps": [
            "numerically_correct_qwen_kernel_bodies",
            "full_qwen_decode_loop_execution",
            "viewer_result_import",
        ],
    }


__all__ = [
    "CALLABLES",
    "build_live_microdecode_plan",
    "repo_relative",
    "write_json",
]
