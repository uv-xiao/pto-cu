"""Compose Qwen resource owners into decode-loop submission plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_decode_loop_runner_impl.bridge_contracts import (
    cuda_live_bridge_contract,
    unit_math_live_bridge_contract,
)
from qwen_decode_loop_runner_impl.graph_materialization import (
    graph_materialization_contract,
)
from qwen_decode_loop_runner_impl.resources import build_resources
from qwen_decode_loop_runner_impl.submission import submission_descriptor_contract


ROOT = Path(__file__).resolve().parents[3]

OWNER_LIFETIME_ORDER = [
    "open_token_pointer_table",
    "open_kv_cache",
    "open_resident_weight_table",
    "materialize_decode_args",
    "materialize_weight_args",
    "submit_persistent_dag",
    "close_resident_weight_table",
    "close_kv_cache",
    "close_token_pointer_table",
]
TASK_ARGUMENT_FIELDS = {
    "a": "input_ids",
    "b": "attention_mask",
    "out": "output_ids",
    "c": "key_cache",
    "d": "value_cache",
    "tensor_args": "resident_weight_tensors",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def decode_arg_records(token_lifecycle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["workload_id"]: item
        for item in token_lifecycle.get("decode_args", {}).get(
            "workload_decode_args",
            [],
        )
    }


def kv_records(kv_lifecycle: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (item["workload_id"], int(item["batch_size"])): item
        for item in kv_lifecycle.get("kv_cache_bindings", [])
    }


def submission_plan(
    *,
    decode_record: dict[str, Any],
    kv_record: dict[str, Any],
    resident_lifecycle: dict[str, Any],
) -> dict[str, Any]:
    scalars = decode_record["scalar_fields"]
    workload_id = decode_record["workload_id"]
    return {
        "workload_id": workload_id,
        "status": "submission_plan_ready",
        "max_batch_size": int(scalars["rows"]),
        "decode_steps": int(scalars["cols"]),
        "first_decode_position": int(scalars["inner"]),
        "owner_lifetime_order": OWNER_LIFETIME_ORDER,
        "task_argument_fields": TASK_ARGUMENT_FIELDS,
        "token_pointer_fields": decode_record["pointer_bindings"],
        "kv_pointer_fields": {
            "c": kv_record["key_cache"],
            "d": kv_record["value_cache"],
        },
        "resident_weight_task_count": resident_lifecycle.get(
            "materialized_task_count",
            0,
        ),
        "resident_weight_pointer_count": resident_lifecycle.get(
            "bound_tensor_pointer_count",
            0,
        ),
        "output_token_accounting": {
            "output_buffer": "output_ids",
            "start_position": int(scalars["inner"]),
            "planned_tokens": int(scalars["cols"]),
            "eos_policy": "planned_stop_after_decode_tokens_or_eos",
        },
    }


def build_submission_plans(
    *,
    token_lifecycle: dict[str, Any],
    kv_lifecycle: dict[str, Any],
    resident_lifecycle: dict[str, Any],
) -> list[dict[str, Any]]:
    decode_records = decode_arg_records(token_lifecycle)
    kv_by_workload_batch = kv_records(kv_lifecycle)
    plans = []
    for workload_id, decode_record in decode_records.items():
        batch = int(decode_record["scalar_fields"]["rows"])
        kv_record = kv_by_workload_batch[(workload_id, batch)]
        plans.append(
            submission_plan(
                decode_record=decode_record,
                kv_record=kv_record,
                resident_lifecycle=resident_lifecycle,
            )
        )
    return plans


def build_decode_loop_runner(
    *,
    mode: str = "offline",
    cache_dir: Path | None = None,
    unit_math_live_payload: dict[str, Any] | None = None,
    token_cuda_live: bool = False,
    kv_cuda_live: bool = False,
    resident_cuda_live: bool = False,
    device: int = 0,
    host_runtime: Path | None = None,
    submission_smoke_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_lifecycle, kv_lifecycle, resident_lifecycle = build_resources(
        mode=mode,
        cache_dir=cache_dir,
        token_cuda_live=token_cuda_live,
        kv_cuda_live=kv_cuda_live,
        resident_cuda_live=resident_cuda_live,
        device=device,
        host_runtime=host_runtime,
    )
    plans = build_submission_plans(
        token_lifecycle=token_lifecycle,
        kv_lifecycle=kv_lifecycle,
        resident_lifecycle=resident_lifecycle,
    )
    resource_modes = {
        "token_pointer_table": token_lifecycle.get("mode", "unknown"),
        "kv_cache": kv_lifecycle.get("mode", "unknown"),
        "resident_weight_table": resident_lifecycle.get("mode", "unknown"),
    }
    live_owners = [
        name
        for name, resource_mode in resource_modes.items()
        if resource_mode == "cuda_live"
    ]
    implemented_contracts = [
        "decode_loop_owner_lifetime_order",
        "persistent_dag_submission_plan",
        "output_token_accounting_plan",
        "cuda_live_resource_bridge_contract",
        "qwen_unit_math_live_bridge_contract",
    ]
    if token_lifecycle.get("mode") == "cuda_live":
        implemented_contracts.append("cuda_live_token_pointer_table_in_runner")
    if kv_lifecycle.get("mode") == "cuda_live":
        implemented_contracts.append("cuda_live_kv_cache_owner_in_runner")
    if resident_lifecycle.get("mode") == "cuda_live":
        implemented_contracts.append("cuda_live_resident_weight_table_in_runner")
    resource_status = {
        "token_pointer_table": token_lifecycle.get("status"),
        "kv_cache": kv_lifecycle.get("status"),
        "resident_weight_table": resident_lifecycle.get("status"),
    }
    graph_materialization = graph_materialization_contract(
        plans=plans,
        resident_lifecycle=resident_lifecycle,
    )
    submission_descriptors = submission_descriptor_contract(
        plans=plans,
        resource_modes=resource_modes,
        resource_status=resource_status,
        execution_evidence=submission_smoke_payload,
    )
    if submission_descriptors["status"] == "resource_backed_descriptors_ready":
        implemented_contracts.append("qwen_decode_loop_submission_descriptors")
    if submission_smoke_payload is not None:
        implemented_contracts.append("qwen_decode_loop_submission_smoke_execution")
    if graph_materialization["status"] == "resource_backed_graph_materialized":
        implemented_contracts.append("qwen_resource_backed_graph_materialization")
    if all(
        item.get("launch_packet_preflight", {}).get("status")
        == "resource_backed_launch_packet_preflight_ready"
        for item in graph_materialization.get("workloads", [])
    ):
        implemented_contracts.append("qwen_resource_backed_launch_packet_preflight")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_decode_loop_runner",
        "status": "decode_loop_runner_plan_ready",
        "mode": (
            "partial_cuda_live_submission_plan"
            if live_owners
            else "dry_run_submission_plan"
        ),
        "resource_lifecycle_status": resource_status,
        "resource_lifecycle_modes": resource_modes,
        "cuda_live_resource_owners": live_owners,
        "cuda_live_submission_descriptor_contract": submission_descriptors,
        "resource_backed_graph_materialization": graph_materialization,
        "cuda_live_bridge_contract": cuda_live_bridge_contract(),
        "unit_math_live_bridge_contract": unit_math_live_bridge_contract(
            unit_math_live_payload,
        ),
        "dag_submission_plans": plans,
        "total_decode_iterations": sum(item["decode_steps"] for item in plans),
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "numerically_correct_qwen_kernel_bodies",
            "cuda_live_decode_loop_execution",
            "viewer_result_import",
        ],
    }
