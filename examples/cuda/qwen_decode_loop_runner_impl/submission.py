"""Build resource-backed CUDA submission descriptors for Qwen decode."""

from __future__ import annotations

from typing import Any


EXPECTED_LIVE_OWNERS = [
    "token_pointer_table",
    "kv_cache",
    "resident_weight_table",
]
QWEN_TASK_FUNCTIONS = [
    {"func_id": 7100, "callable": "qwen_embedding_lookup"},
    {"func_id": 7101, "callable": "qwen_rmsnorm_input"},
    {"func_id": 7102, "callable": "qwen_attention_qkv"},
    {"func_id": 7103, "callable": "qwen_attention_qk_norm"},
    {"func_id": 7104, "callable": "qwen_attention_o"},
    {"func_id": 7105, "callable": "qwen_rmsnorm_post_attention"},
    {"func_id": 7106, "callable": "qwen_mlp_gate_up"},
    {"func_id": 7107, "callable": "qwen_mlp_down"},
    {"func_id": 7108, "callable": "qwen_final_norm"},
    {"func_id": 7109, "callable": "qwen_logits"},
]


def submission_descriptor_contract(
    *,
    plans: list[dict[str, Any]],
    resource_modes: dict[str, str],
    resource_status: dict[str, Any],
    execution_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    live_ready = all(
        resource_modes.get(owner) == "cuda_live" for owner in EXPECTED_LIVE_OWNERS
    )
    lifecycle_ready = all(
        isinstance(status, str) and status.endswith("_ready")
        for status in resource_status.values()
    )
    descriptors = [descriptor_for_plan(plan) for plan in plans]
    executed = execution_evidence is not None and execution_evidence.get("status") == "pass"
    return {
        "status": (
            "resource_backed_descriptors_ready"
            if live_ready and lifecycle_ready and descriptors
            else "resource_backed_descriptors_incomplete"
        ),
        "runtime": "cuda/persistent_device",
        "entry_name": "pto_persistent_dag_f32_executor",
        "task_body_source": "examples/cuda/qwen_persistent_task_bodies.py",
        "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
        "execution_status": (
            "diagnostic_descriptor_smoke_passed" if executed else "not_executed"
        ),
        "execution_evidence": execution_summary(execution_evidence),
        "resource_preconditions": resource_status,
        "required_cuda_live_owners": EXPECTED_LIVE_OWNERS,
        "descriptors": descriptors,
        "remaining_gap": (
            "resource_backed_full_qwen_decode_loop_execution"
            if executed
            else "run_prepared_full_qwen_decode_loop"
        ),
    }


def execution_summary(evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    counters = evidence.get("scheduler_counters", {})
    timing = evidence.get("timing_ns", {})
    return {
        "kind": evidence.get("kind"),
        "status": evidence.get("status"),
        "serving_coverage": evidence.get("serving_coverage"),
        "func_id_sequence": evidence.get("func_id_sequence", []),
        "completed_count": int(counters.get("completed_count", 0)),
        "error_count": int(counters.get("error_count", 0)),
        "max_abs_error": float(evidence.get("max_abs_error", 0.0)),
        "host_wall_ns": int(timing.get("host_wall", 0)),
        "device_wall_ns": int(timing.get("device_wall", 0)),
    }


def descriptor_for_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_id": plan["workload_id"],
        "status": "descriptor_ready",
        "callable_id": 0,
        "decode_steps": plan["decode_steps"],
        "first_decode_position": plan["first_decode_position"],
        "graph_task_count": plan["resident_weight_task_count"],
        "task_function_count": len(QWEN_TASK_FUNCTIONS),
        "func_id_sequence": [item["func_id"] for item in QWEN_TASK_FUNCTIONS],
        "callables": [item["callable"] for item in QWEN_TASK_FUNCTIONS],
        "task_argument_fields": plan["task_argument_fields"],
        "token_pointer_fields": plan["token_pointer_fields"],
        "kv_pointer_fields": plan["kv_pointer_fields"],
        "resident_weight_pointer_count": plan["resident_weight_pointer_count"],
        "run_prepared_repetitions": plan["decode_steps"],
        "output_token_accounting": plan["output_token_accounting"],
    }
