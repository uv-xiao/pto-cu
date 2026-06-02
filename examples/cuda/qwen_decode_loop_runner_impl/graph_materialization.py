"""Materialize compact Qwen decode-loop graph binding evidence."""

from __future__ import annotations

from typing import Any

from qwen_decode_loop_runner_impl.launch_preflight import (
    launch_packet_preflight,
)
from qwen_decode_loop_runner_impl.submission import QWEN_TASK_FUNCTIONS
from qwen_persistent_weight_materialization import build_materialization_manifest


CALLABLE_FUNC_IDS = {
    item["callable"]: item["func_id"] for item in QWEN_TASK_FUNCTIONS
}


def graph_materialization_contract(
    *,
    plans: list[dict[str, Any]],
    resident_lifecycle: dict[str, Any],
    activation_workspace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    materialization = build_materialization_manifest(
        pointer_table=resident_lifecycle.get("pointer_table"),
    )
    descriptors = materialization.get("materialized_task_descriptors", [])
    ready_descriptors = [
        descriptor for descriptor in descriptors if descriptor.get("status") == "ready"
    ]
    workload_records = [
        workload_materialization(
            plan=plan,
            descriptors=ready_descriptors,
            activation_workspace=activation_workspace,
        )
        for plan in plans
    ]
    ready = (
        materialization.get("status") == "persistent_weight_materialization_ready"
        and bool(workload_records)
        and all(item["status"] == "resource_backed_graph_materialized" for item in workload_records)
    )
    return {
        "status": (
            "resource_backed_graph_materialized"
            if ready
            else "resource_backed_graph_incomplete"
        ),
        "runtime": "cuda/persistent_device",
        "task_struct": materialization.get("abi", {}).get("task_struct"),
        "task_struct_size_bytes": materialization.get("abi", {}).get("sizeof_bytes"),
        "materialization_status": materialization.get("status"),
        "materialized_task_count": len(ready_descriptors),
        "bound_tensor_pointer_count": materialization.get("bound_tensor_pointer_count", 0),
        "missing_pointer_count": materialization.get("missing_pointer_count", 0),
        "workloads": workload_records,
        "remaining_gap": "run_prepared_resource_backed_decode_loop",
    }


def workload_materialization(
    *,
    plan: dict[str, Any],
    descriptors: list[dict[str, Any]],
    activation_workspace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_fields = ready_token_fields(plan.get("token_pointer_fields", []))
    kv_fields = ready_kv_fields(plan.get("kv_pointer_fields", {}))
    ready = (
        len(descriptors) == plan.get("resident_weight_task_count")
        and len(token_fields) == 3
        and len(kv_fields) == 2
    )
    return {
        "workload_id": plan["workload_id"],
        "status": (
            "resource_backed_graph_materialized"
            if ready
            else "resource_backed_graph_incomplete"
        ),
        "decode_steps": plan["decode_steps"],
        "graph_task_count": len(descriptors),
        "token_pointer_fields": token_fields,
        "kv_pointer_fields": kv_fields,
        "first_task": task_summary(descriptors[0]) if descriptors else None,
        "last_task": task_summary(descriptors[-1]) if descriptors else None,
        "sample_layer_tasks": [
            task_summary(descriptor)
            for descriptor in descriptors
            if descriptor.get("id")
            in {
                "layer_0_attention_qkv",
                "layer_0_mlp_gate_up",
                "layer_35_attention_qkv",
                "layer_35_mlp_down",
            }
        ],
        "launch_packet_preflight": launch_packet_preflight(
            plan=plan,
            descriptors=descriptors,
            activation_workspace=activation_workspace,
        ),
    }


def ready_token_fields(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "field": item["field"],
            "buffer": item["buffer"],
            "device_ptr_hex": item["device_ptr_hex"],
            "byte_count": item["byte_count"],
        }
        for item in bindings
        if isinstance(item, dict) and item.get("device_ptr_hex")
    ]


def ready_kv_fields(bindings: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for field, item in sorted(bindings.items()):
        if not isinstance(item, dict) or not item.get("device_ptr_hex"):
            continue
        records.append(
            {
                "field": field,
                "cache": item["cache"],
                "device_ptr_hex": item["device_ptr_hex"],
                "byte_count": item["byte_count"],
                "element_dtype": item["element_dtype"],
            }
        )
    return records


def task_summary(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": descriptor["id"],
        "callable": descriptor["callable"],
        "func_id": CALLABLE_FUNC_IDS[descriptor["callable"]],
        "tensor_arg_count": descriptor["tensor_arg_count"],
        "tensor_arg_slots": [
            tensor_arg_summary(arg)
            for arg in descriptor["tensor_args"]
        ],
    }


def tensor_arg_summary(arg: dict[str, Any]) -> dict[str, Any]:
    return {
        key: arg[key]
        for key in (
            "arg",
            "slot_id",
            "tensor",
            "role",
            "status",
            "device_ptr_source",
            "device_ptr_hex",
        )
        if key in arg
    }
