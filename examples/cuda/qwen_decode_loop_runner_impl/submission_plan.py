"""Qwen decode-loop DAG submission plan helpers."""

from __future__ import annotations

from typing import Any


OWNER_LIFETIME_ORDER = [
    "open_token_pointer_table",
    "open_kv_cache",
    "open_resident_weight_table",
    "open_activation_workspace",
    "materialize_decode_args",
    "materialize_weight_args",
    "submit_persistent_dag",
    "close_activation_workspace",
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
    return {
        "workload_id": decode_record["workload_id"],
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
        "task_shape_fields": {
            "b_batch_stride": int(kv_record["sequence_capacity_tokens"]),
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
        plans.append(
            submission_plan(
                decode_record=decode_record,
                kv_record=kv_by_workload_batch[(workload_id, batch)],
                resident_lifecycle=resident_lifecycle,
            )
        )
    return plans
