"""Reviewable plan for Qwen unit-math live CUDA execution."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from qwen_persistent_proxy_live_impl.plan import repo_relative, write_json
from qwen_persistent_task_bodies_impl.lifecycle import FUNC_ID_BASE
from qwen_persistent_task_bodies_impl.oracle import build_qwen_unit_math_oracle


CALLABLES = [
    ("qwen_rmsnorm_input", FUNC_ID_BASE + 1),
    ("qwen_attention_qkv", FUNC_ID_BASE + 2),
    ("qwen_mlp_gate_up", FUNC_ID_BASE + 6),
    ("qwen_final_norm", FUNC_ID_BASE + 8),
    ("qwen_logits", FUNC_ID_BASE + 9),
]


def _round(values: list[float]) -> list[float]:
    return [round(value, 6) for value in values]


def _rmsnorm(hidden: list[float], weight: list[float]) -> list[float]:
    mean_square = sum(value * value for value in hidden) / len(hidden)
    scale = 1.0 / math.sqrt(mean_square + 1e-6)
    return [value * scale * weight[index] for index, value in enumerate(hidden)]


def _unit_expected(hidden: list[float], inputs: dict[str, Any]) -> dict[str, list[float]]:
    rmsnorm = _rmsnorm(hidden, inputs["norm_weight"])
    key_cache = [
        rmsnorm[index] * inputs["k_proj_weight"][index]
        for index in range(4)
    ]
    value_cache = [
        rmsnorm[index] * inputs["v_proj_weight"][index]
        for index in range(4)
    ]
    silu_gate = [
        gate / (1.0 + math.exp(-gate))
        for gate in inputs["gate_proj_weight"]
    ]
    mlp = [
        silu_gate[index] * inputs["up_proj_weight"][index]
        for index in range(4)
    ]
    final_norm = _rmsnorm(mlp, inputs["final_norm_weight"])
    logits = [
        final_norm[index] * inputs["lm_head_weight"][index]
        for index in range(4)
    ]
    return {
        "rmsnorm": _round(rmsnorm),
        "attention_context": _round(value_cache),
        "key_cache": _round(key_cache),
        "value_cache": _round(value_cache),
        "mlp_swiglu": _round(mlp),
        "final_norm": _round(final_norm),
        "logits": _round(logits),
    }


def _decode_iterations(
    *,
    repeat_runs: int,
    initial_hidden: list[float],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    if repeat_runs < 1:
        raise ValueError("repeat_runs must be positive")
    hidden = list(initial_hidden)
    iterations = []
    for index in range(repeat_runs):
        expected = _unit_expected(hidden, inputs)
        iterations.append(
            {
                "iteration": index,
                "inputs": {"hidden": _round(hidden)},
                "expected": expected,
            }
        )
        hidden = list(expected["logits"])
    return iterations


def build_unit_math_live_plan(
    *,
    repeat_runs: int = 1,
    scheduler_blocks: int = 1,
    worker_blocks: int = 1,
    queue_capacity: int = 8,
    block_dim: int = 128,
) -> dict[str, Any]:
    oracle = build_qwen_unit_math_oracle()
    steps = oracle["steps"]
    inputs = {
        "hidden": oracle["inputs"]["hidden"],
        "norm_weight": oracle["inputs"]["norm_weight"],
        "q_proj_weight": [0.5, 0.5, 0.5, 0.5],
        "k_proj_weight": [0.25, 0.25, 0.25, 0.25],
        "v_proj_weight": [0.4, 0.3, 0.3, 0.2],
        "gate_proj_weight": steps["gate_projection"],
        "up_proj_weight": steps["up_projection"],
        "final_norm_weight": [1.0, 1.0, 1.0, 1.0],
        "lm_head_weight": [3.4, 4.4, 5.0, 6.0],
    }
    decode_iterations = _decode_iterations(
        repeat_runs=repeat_runs,
        initial_hidden=inputs["hidden"],
        inputs=inputs,
    )
    return {
        "kind": "pto_qwen_unit_math_live_execution_plan",
        "status": "ready_to_run",
        "scope": oracle["scope"],
        "model_id": "Qwen/Qwen3-8B",
        "runtime": "cuda/persistent_device",
        "tasks": [name for name, _ in CALLABLES],
        "dag": {
            "task_count": len(CALLABLES),
            "dependent_count": len(CALLABLES) - 1,
            "scheduler_blocks": scheduler_blocks,
            "worker_blocks": worker_blocks,
            "queue_capacity": queue_capacity,
            "block_dim": block_dim,
            "dependency_edges": [
                ["qwen_rmsnorm_input", "qwen_attention_qkv"],
                ["qwen_attention_qkv", "qwen_mlp_gate_up"],
                ["qwen_mlp_gate_up", "qwen_final_norm"],
                ["qwen_final_norm", "qwen_logits"],
            ],
        },
        "decode_loop": {
            "repeat_runs": repeat_runs,
            "planned_task_executions": repeat_runs * len(CALLABLES),
            "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
            "reset_between_runs": [
                "fanin",
                "ready_flags",
                "completion_flags",
                "counters",
                "unit_outputs",
            ],
            "carried_between_runs": [
                "hidden_state_from_previous_logits",
                "weight_buffers",
                "kv_cache_buffers",
            ],
        },
        "decode_iterations": decode_iterations,
        "inputs": inputs,
        "expected": decode_iterations[0]["expected"],
        "implemented_contracts": [
            "qwen_unit_math_cuda_live_execution_plan",
            "persistent_device_unit_math_dag_launch_plan",
            "qwen_unit_math_source_coverage",
            "qwen_unit_math_final_norm_live_execution",
        ],
        "remaining_runtime_gaps": [
            "full_qwen_decode_loop_execution",
            "viewer_result_import",
        ],
    }


__all__ = [
    "CALLABLES",
    "build_unit_math_live_plan",
    "repo_relative",
    "write_json",
]
