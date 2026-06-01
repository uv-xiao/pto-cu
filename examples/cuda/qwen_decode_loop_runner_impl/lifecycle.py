"""Compose Qwen resource owners into decode-loop submission plans."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TOKEN_POINTER = ROOT / "examples" / "cuda" / "qwen_token_pointer_table.py"
KV_CACHE = ROOT / "examples" / "cuda" / "qwen_kv_cache_binding.py"
RESIDENT_WEIGHTS = ROOT / "examples" / "cuda" / "qwen_resident_weight_table.py"
LIVE_MICRODECODE_ARTIFACT = (
    "tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/"
    "qwen-microdecode-loop.json"
)
LIVE_UNIT_MATH_ARTIFACT = (
    "tmp/cuda-backend/pto-serving-unit-math-live-2026-06-01/"
    "qwen-unit-math-live.json"
)

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
LIVE_MICRODECODE_FIELDS = {
    "a": "hidden_state",
    "b": "attention_mask",
    "out": "logits_out",
    "c": "key_cache_mutable",
    "d": "value_cache_mutable",
    "tensor_args": "resident_weight_tensors",
}
LIVE_DECODE_LOOP_REUSE = {
    "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
    "reset_between_runs": [
        "fanin",
        "ready_flags",
        "completion_flags",
        "counters",
    ],
    "carried_between_runs": ["key_cache_mutable", "value_cache_mutable"],
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_resources(
    *,
    mode: str,
    cache_dir: Path | None,
    token_cuda_live: bool = False,
    device: int = 0,
    host_runtime: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    token_module = load_module(TOKEN_POINTER, "qwen_token_pointer_for_runner")
    kv_module = load_module(KV_CACHE, "qwen_kv_cache_for_runner")
    resident_module = load_module(RESIDENT_WEIGHTS, "qwen_resident_weights_for_runner")
    token_kwargs: dict[str, Any] = {
        "mode": mode,
        "cache_dir": cache_dir,
        "cuda_live": token_cuda_live,
        "device": device,
    }
    if host_runtime is not None:
        token_kwargs["host_runtime"] = host_runtime
    return (
        token_module.build_token_pointer_table_lifecycle(**token_kwargs),
        kv_module.build_kv_cache_lifecycle(),
        resident_module.build_resident_table_lifecycle(),
    )


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


def cuda_live_bridge_contract() -> dict[str, Any]:
    return {
        "status": "diagnostic_bridge_ready",
        "runtime": "cuda/persistent_device",
        "source_live_artifact": LIVE_MICRODECODE_ARTIFACT,
        "submission_to_live_fields": LIVE_MICRODECODE_FIELDS,
        "decode_loop_reuse": LIVE_DECODE_LOOP_REUSE,
        "serving_coverage": "diagnostic_microdecode",
        "remaining_gap": "full_qwen_decode_loop_execution",
    }


def unit_math_live_bridge_contract(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "status": "diagnostic_bridge_ready",
        "runtime": "cuda/persistent_device",
        "source_live_artifact": LIVE_UNIT_MATH_ARTIFACT,
        "submission_to_live_fields": {
            "a": "hidden_state",
            "out": "logits_out",
            "c": "key_cache_mutable",
            "d": "value_cache_mutable",
            "tensor_args": "unit_weight_tensors",
        },
        "decode_loop_reuse": {
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
        "serving_coverage": "diagnostic_unit_math",
        "remaining_gap": "full_qwen_decode_loop_execution",
    }
    if payload is None:
        return contract
    summary = payload.get("decode_loop_summary", {})
    contract["status"] = "diagnostic_bridge_executed"
    contract["live_summary"] = {
        "status": payload.get("status", "unknown"),
        "repeat_runs": int(summary.get("repeat_runs", 1)),
        "total_completed_count": int(summary.get("total_completed_count", 0)),
        "total_error_count": int(summary.get("total_error_count", 0)),
        "max_abs_error": float(payload.get("max_abs_error", 0.0)),
    }
    return contract


def build_decode_loop_runner(
    *,
    mode: str = "offline",
    cache_dir: Path | None = None,
    unit_math_live_payload: dict[str, Any] | None = None,
    token_cuda_live: bool = False,
    device: int = 0,
    host_runtime: Path | None = None,
) -> dict[str, Any]:
    token_lifecycle, kv_lifecycle, resident_lifecycle = build_resources(
        mode=mode,
        cache_dir=cache_dir,
        token_cuda_live=token_cuda_live,
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
    return {
        "schema_version": 1,
        "kind": "pto_qwen_decode_loop_runner",
        "status": "decode_loop_runner_plan_ready",
        "mode": (
            "partial_cuda_live_submission_plan"
            if live_owners
            else "dry_run_submission_plan"
        ),
        "resource_lifecycle_status": {
            "token_pointer_table": token_lifecycle.get("status"),
            "kv_cache": kv_lifecycle.get("status"),
            "resident_weight_table": resident_lifecycle.get("status"),
        },
        "resource_lifecycle_modes": resource_modes,
        "cuda_live_resource_owners": live_owners,
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
