"""Reviewable plan for the controlled Qwen QKV live proxy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_persistent_task_bodies_impl.lifecycle import FUNC_ID_BASE, ROOT
from qwen_persistent_task_bodies_impl.oracle import build_numeric_oracle


CALLABLE_NAME = "qwen_attention_qkv"
FUNC_ID = FUNC_ID_BASE + 2


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _qkv_oracle() -> dict[str, Any]:
    oracle = build_numeric_oracle([CALLABLE_NAME])
    return oracle["sample_outputs"][0]


def build_live_proxy_plan(
    *,
    scheduler_blocks: int = 1,
    worker_blocks: int = 1,
    queue_capacity: int = 4,
    block_dim: int = 128,
) -> dict[str, Any]:
    qkv = _qkv_oracle()
    return {
        "kind": "pto_qwen_proxy_live_execution_plan",
        "status": "ready_to_run",
        "scope": "controlled_proxy_not_full_qwen",
        "model_id": "Qwen/Qwen3-8B",
        "runtime": "cuda/persistent_device",
        "callable": CALLABLE_NAME,
        "func_id": FUNC_ID,
        "dag": {
            "task_count": 1,
            "dependent_count": 0,
            "scheduler_blocks": scheduler_blocks,
            "worker_blocks": worker_blocks,
            "queue_capacity": queue_capacity,
            "block_dim": block_dim,
        },
        "task_argument_fields": {
            "a": "hidden_state",
            "b": "attention_mask",
            "out": "attention_output",
            "c": "key_cache_mutable",
            "d": "value_cache_mutable",
            "tensor_args[0]": "q_proj_weight",
        },
        "inputs": {
            "a": [10.0, 11.0, 12.0, 13.0],
            "b": [1.0, 1.0, 1.0, 1.0],
            "c": [0.5, 0.5, 0.5, 0.5],
            "d": [0.5, 0.5, 0.5, 0.5],
            "weights": [[1.0, 2.0, 3.0, 4.0]],
        },
        "expected": {
            "out": qkv["expected_out"],
            "c": qkv["expected_c"],
            "d": qkv["expected_d"],
        },
        "implemented_contracts": [
            "controlled_proxy_live_cuda_execution_plan",
            "persistent_device_single_task_dag_launch_plan",
            "qwen_attention_qkv_mutable_kv_live_contract",
        ],
        "remaining_runtime_gaps": [
            "numerically_correct_qwen_kernel_bodies",
            "full_qwen_decode_loop_execution",
            "viewer_result_import",
        ],
    }

