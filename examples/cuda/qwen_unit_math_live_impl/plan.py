"""Reviewable plan for Qwen unit-math live CUDA execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_persistent_proxy_live_impl.plan import repo_relative, write_json
from qwen_persistent_task_bodies_impl.lifecycle import FUNC_ID_BASE
from qwen_persistent_task_bodies_impl.oracle import build_qwen_unit_math_oracle


CALLABLES = [
    ("qwen_rmsnorm_input", FUNC_ID_BASE + 1),
    ("qwen_attention_qkv", FUNC_ID_BASE + 2),
    ("qwen_mlp_gate_up", FUNC_ID_BASE + 6),
    ("qwen_logits", FUNC_ID_BASE + 9),
]


def build_unit_math_live_plan(
    *,
    scheduler_blocks: int = 1,
    worker_blocks: int = 1,
    queue_capacity: int = 8,
    block_dim: int = 128,
) -> dict[str, Any]:
    oracle = build_qwen_unit_math_oracle()
    steps = oracle["steps"]
    return {
        "kind": "pto_qwen_unit_math_live_execution_plan",
        "status": "ready_to_run",
        "scope": oracle["scope"],
        "model_id": "Qwen/Qwen3-8B",
        "runtime": "cuda/persistent_device",
        "tasks": [name for name, _ in CALLABLES],
        "dag": {
            "task_count": 4,
            "dependent_count": 3,
            "scheduler_blocks": scheduler_blocks,
            "worker_blocks": worker_blocks,
            "queue_capacity": queue_capacity,
            "block_dim": block_dim,
            "dependency_edges": [
                ["qwen_rmsnorm_input", "qwen_attention_qkv"],
                ["qwen_attention_qkv", "qwen_mlp_gate_up"],
                ["qwen_mlp_gate_up", "qwen_logits"],
            ],
        },
        "inputs": {
            "hidden": oracle["inputs"]["hidden"],
            "norm_weight": oracle["inputs"]["norm_weight"],
            "q_proj_weight": [0.5, 0.5, 0.5, 0.5],
            "k_proj_weight": [0.25, 0.25, 0.25, 0.25],
            "v_proj_weight": [0.4, 0.3, 0.3, 0.2],
            "gate_proj_weight": steps["gate_projection"],
            "up_proj_weight": steps["up_projection"],
            "lm_head_weight": [3.4, 4.4, 5.0, 6.0],
        },
        "expected": {
            "rmsnorm": steps["rmsnorm_input"],
            "attention_context": steps["attention_context"],
            "key_cache": steps["key_cache_after"],
            "value_cache": steps["value_cache_after"],
            "mlp_swiglu": steps["mlp_swiglu"],
            "logits": steps["logits"],
        },
        "implemented_contracts": [
            "qwen_unit_math_cuda_live_execution_plan",
            "persistent_device_unit_math_dag_launch_plan",
            "qwen_unit_math_source_coverage",
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
