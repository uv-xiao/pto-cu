"""Result assembly for resource-backed Qwen diagnostic execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_decode_loop_runner_impl.decode_feedback import feedback_summary
from qwen_decode_loop_runner_impl.launch_helpers import numeric_task_mode_summary
from qwen_decode_loop_runner_impl.resource_check_policy import (
    logits_check_summary,
    logits_summary_stable_for_checked_steps,
)
from qwen_decode_loop_runner_impl.resource_execution_policy import (
    decode_step_execution_summary,
    implemented_contracts,
    logits_stability_key,
    logits_summary_stable,
)
from qwen_persistent_proxy_live_impl.runtime import device_name


QWEN_FUNC_ID_BY_CALLABLE = {
    "qwen_embedding_lookup": 7100,
    "qwen_rmsnorm_input": 7101,
    "qwen_attention_qkv": 7102,
    "qwen_attention_qk_norm": 7103,
    "qwen_attention_o": 7104,
    "qwen_rmsnorm_post_attention": 7105,
    "qwen_mlp_gate_up": 7106,
    "qwen_mlp_down": 7107,
    "qwen_final_norm": 7108,
    "qwen_logits": 7109,
}


def build_execution_result(
    *,
    session: Any,
    arch: str,
    prepared: Any,
    descriptors: list[dict[str, Any]],
    workload_results: list[dict[str, Any]],
    repeat_runs: int,
    decode_step_limit: int | None,
    workload_ids: list[str] | None,
    max_task_count: int | None,
    task_selection: str,
    scheduler_blocks: int,
    worker_blocks: int,
    grid_dim: int,
    logits_check_policy: str,
    logits_active_cols_policy: dict[str, Any],
    numeric_task_mode: str,
    prefill_prompt: bool,
    repo_relative,
) -> dict[str, Any]:
    passed = workload_results and all(
        item["status"] == "pass" for item in workload_results
    )
    decode_step_execution = decode_step_execution_summary(
        workload_results,
        decode_step_limit=decode_step_limit,
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_resource_backed_execution",
        "status": "pass" if passed else "fail",
        "runtime": "cuda/persistent_device",
        "serving_coverage": "diagnostic_resource_backed_qwen_dag",
        "device": {
            "ordinal": session.device,
            "name": device_name(session.device),
            "arch": arch,
        },
        "artifact": artifact_summary(prepared, repo_relative),
        "context_policy": "one_cuda_context_for_all_resource_owners",
        "repeat_policy": {
            "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
            "repeat_runs_per_workload": max(1, int(repeat_runs)),
            "decode_step_limit": decode_step_limit,
            "workload_filter": workload_ids or "all",
            "max_task_count": max_task_count,
            "task_selection": task_selection,
            "scheduler_blocks": int(scheduler_blocks),
            "worker_blocks": int(worker_blocks),
            "grid_dim": int(grid_dim),
            "logits_check_policy": logits_check_policy,
            "logits_active_cols_policy": logits_active_cols_policy,
            "numeric_task_mode": numeric_task_mode_summary(numeric_task_mode),
            "graph_state_policy": "fresh_graph_state_per_repeat",
            "prompt_prefill": bool(prefill_prompt),
        },
        "task_coverage": task_coverage(descriptors),
        "decode_step_execution": decode_step_execution,
        "workloads": workload_results,
        "implemented_contracts": execution_contracts(
            decode_step_limit=decode_step_limit,
            workload_results=workload_results,
            numeric_task_mode=numeric_task_mode,
            decode_step_execution=decode_step_execution,
        ),
        "remaining_runtime_gaps": [
            "full_qwen_numerical_correctness",
            "full_serving_viewer_result_import",
        ],
}


def task_coverage(descriptors: list[dict[str, Any]]) -> dict[str, Any]:
    callables = [
        item["callable"]
        for item in descriptors
        if "callable" in item
    ]
    func_ids = []
    for item in descriptors:
        func_id = task_func_id(item)
        if func_id is not None:
            func_ids.append(func_id)
    return {
        "task_count": len(descriptors),
        "func_id_sequence": func_ids,
        "callables": callables,
    }


def task_func_id(descriptor: dict[str, Any]) -> int | None:
    if "func_id" in descriptor:
        return int(descriptor["func_id"])
    callable_name = descriptor.get("callable")
    if callable_name not in QWEN_FUNC_ID_BY_CALLABLE:
        return None
    return QWEN_FUNC_ID_BY_CALLABLE[callable_name]


def build_workload_result(
    *,
    plan: dict[str, Any],
    packet_len: int,
    repeat_results: list[dict[str, Any]],
    decode_step_limit: int | None,
    logits_check_policy: str,
    numeric_task_mode: str,
    prefill_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prefill_results = prefill_results or []
    last = repeat_results[-1]
    return {
        "workload_id": plan["workload_id"],
        "status": "pass"
        if all_success(prefill_results) and all_success(repeat_results)
        else "fail",
        "run_prepared_status": int(last["run_prepared_status"]),
        "repeat_runs": len(repeat_results),
        "planned_decode_steps": int(plan["decode_steps"]),
        "executed_decode_steps": (
            len(repeat_results) if decode_step_limit is not None else 0
        ),
        "decode_step_limit": decode_step_limit,
        "prompt_prefill": prompt_prefill_summary(
            plan=plan,
            prefill_results=prefill_results,
        ),
        "execution_mode": (
            "bounded_decode_steps"
            if decode_step_limit is not None
            else "repeat_submissions"
        ),
        "logits_check_policy": logits_check_policy,
        "numeric_task_mode": numeric_task_mode_summary(numeric_task_mode),
        "logits_check_summary": logits_check_summary(repeat_results),
        "repeat_results": repeat_results,
        "prefill_results": prefill_results,
        "graph_task_count": packet_len,
        "scheduler_counters": last["scheduler_counters"],
        "total_completed_count": total_counter(repeat_results, "completed_count"),
        "total_error_count": total_counter(repeat_results, "error_count"),
        "output_sample": last["output_sample"],
        "logits_summary": last["logits_summary"],
        "logits_summary_stable": checked_logits_stable(
            repeat_results,
            logits_check_policy,
        ),
        "decode_feedback": feedback_summary(repeat_results),
        "timing_ns": total_timing(repeat_results),
    }


def all_success(repeat_results: list[dict[str, Any]]) -> bool:
    if not repeat_results:
        return True
    return all(item["status"] == "pass" for item in repeat_results)


def prompt_prefill_summary(
    *,
    plan: dict[str, Any],
    prefill_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not prefill_results:
        return {"status": "not_requested"}
    expected = int(plan.get("active_prompt_tokens", 0))
    executed = len(prefill_results)
    return {
        "status": "prompt_prefill_executed"
        if executed == expected and all_success(prefill_results)
        else "prompt_prefill_incomplete",
        "expected_prompt_positions": expected,
        "executed_prompt_positions": executed,
        "first_decode_position": int(plan.get("first_decode_position", 0)),
        "total_completed_count": total_counter(prefill_results, "completed_count"),
        "total_error_count": total_counter(prefill_results, "error_count"),
    }


def total_counter(repeat_results: list[dict[str, Any]], name: str) -> int:
    return sum(item["scheduler_counters"][name] for item in repeat_results)


def total_timing(repeat_results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "host_wall": sum(item["timing_ns"]["host_wall"] for item in repeat_results),
        "device_wall": sum(item["timing_ns"]["device_wall"] for item in repeat_results),
    }


def checked_logits_stable(
    repeat_results: list[dict[str, Any]],
    logits_check_policy: str,
) -> bool:
    if logits_check_policy == "every_step":
        return logits_summary_stable(repeat_results)
    return logits_summary_stable_for_checked_steps(
        repeat_results,
        logits_stability_key,
    )


def artifact_summary(prepared: Any, repo_relative) -> dict[str, Any]:
    artifact = prepared.artifact
    return {
        "cache_key": artifact.cache_key,
        "cache_hit": artifact.cache_hit,
        "source_path": repo_relative(Path(artifact.source_path)),
        "ptx_path": repo_relative(Path(artifact.ptx_path)),
        "entry_name": artifact.entry_name,
        "source_kind": artifact.source_kind,
    }


def decode_feedback_contract_status(workload_results: list[dict[str, Any]]) -> str:
    statuses = {
        item.get("decode_feedback", {}).get("status", "not_requested")
        for item in workload_results
    }
    if statuses == {"device_token_feedback_observed"}:
        return "device_token_feedback_observed"
    if statuses == {"diagnostic_token_feedback_applied"}:
        return "diagnostic_token_feedback_applied"
    return "not_requested"


def execution_contracts(
    *,
    decode_step_limit: int | None,
    workload_results: list[dict[str, Any]],
    numeric_task_mode: str,
    decode_step_execution: dict[str, Any],
) -> list[str]:
    contracts = implemented_contracts(
        decode_step_limit,
        token_feedback_status=decode_feedback_contract_status(workload_results),
        policy_length_complete=bool(
            decode_step_execution.get("policy_length_complete"),
        ),
    )
    if numeric_task_mode in {"unit_math", "unit_math_full_rmsnorm"}:
        contracts.append("qwen_resource_backed_unit_numeric_task_mode")
        if numeric_task_mode == "unit_math":
            contracts.append("qwen_resource_backed_external_rmsnorm_scale")
        if numeric_task_mode == "unit_math_full_rmsnorm":
            contracts.append("qwen_resource_backed_full_rmsnorm_reduction")
        contracts.append("qwen_resource_backed_weighted_elementwise_branches")
    if dynamic_rope_refresh_ready(workload_results):
        contracts.append("qwen_dynamic_rope_table_refresh")
    return contracts


def dynamic_rope_refresh_ready(workload_results: list[dict[str, Any]]) -> bool:
    refreshes = [
        repeat.get("rope_table_refresh", {})
        for workload in workload_results
        for repeat in workload.get("repeat_results", [])
        if repeat.get("decode_step_index") is not None
    ]
    return bool(refreshes) and all(
        item.get("status") == "refreshed"
        and item.get("policy") == "position_correct_for_decode_step"
        for item in refreshes
    )
