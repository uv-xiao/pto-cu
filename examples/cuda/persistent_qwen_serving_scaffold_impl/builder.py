"""Build the Qwen persistent serving scaffold artifact."""

from __future__ import annotations

from typing import Any

from .loaders import load_artifacts
from .stages import build_stages, serving_workload_contracts
from .status import readiness


def build_scaffold() -> dict[str, Any]:
    artifacts = load_artifacts()
    flags = readiness(artifacts)
    stages = build_stages(artifacts, flags)
    missing = [
        item
        for item in stages
        if item["required_for_full_serving"] and item["status"] != "pass"
    ]
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_serving_scaffold",
        "status": "partial" if missing else "pass",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "runtime": "cuda/persistent_device",
        "serving_workloads": serving_workload_contracts(),
        "lifecycle_plan": artifacts["lifecycle_plan"],
        "prompt_accounting": artifacts["prompt_accounting"],
        "runtime_input_binding": artifacts["runtime_input_binding"],
        "cuda_token_buffer_binding": artifacts["cuda_token_buffer_binding"],
        "persistent_decode_args": artifacts["persistent_decode_args"],
        "token_pointer_table": artifacts["token_pointer_table"],
        "weight_inventory": artifacts["weight_inventory"],
        "safetensors_shards": artifacts["safetensors_shards"],
        "safetensors_metadata": artifacts["safetensors_metadata"],
        "cuda_weight_binding": artifacts["cuda_weight_binding"],
        "persistent_weight_args": artifacts["persistent_weight_args"],
        "persistent_weight_materialization": artifacts["persistent_weight_materialization"],
        "resident_weight_table": artifacts["resident_weight_table"],
        "kv_cache_binding": artifacts["kv_cache_binding"],
        "decode_loop_runner": artifacts["decode_loop_runner"],
        "persistent_task_bodies": artifacts["task_bodies"],
        "stages": stages,
        "missing_stage_ids": [item["id"] for item in missing],
        "next_action": (
            "Implement the missing PTO serving host/runtime stages, then "
            "import Qwen/Qwen3-8B full-serving rows."
        ),
    }
