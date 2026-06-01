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


def build_execution_result(
    *,
    session: Any,
    arch: str,
    prepared: Any,
    workload_results: list[dict[str, Any]],
    repeat_runs: int,
    decode_step_limit: int | None,
    workload_ids: list[str] | None,
    logits_check_policy: str,
    numeric_task_mode: str,
    repo_relative,
) -> dict[str, Any]:
    passed = workload_results and all(
        item["status"] == "pass" for item in workload_results
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
            "logits_check_policy": logits_check_policy,
            "numeric_task_mode": numeric_task_mode_summary(numeric_task_mode),
            "graph_state_policy": "fresh_graph_state_per_repeat",
        },
        "decode_step_execution": decode_step_execution_summary(
            workload_results,
            decode_step_limit=decode_step_limit,
        ),
        "workloads": workload_results,
        "implemented_contracts": execution_contracts(
            decode_step_limit=decode_step_limit,
            workload_results=workload_results,
            numeric_task_mode=numeric_task_mode,
        ),
        "remaining_runtime_gaps": [
            "full_qwen_numerical_correctness",
            "full_serving_viewer_result_import",
        ],
    }


def build_workload_result(
    *,
    plan: dict[str, Any],
    packet_len: int,
    repeat_results: list[dict[str, Any]],
    decode_step_limit: int | None,
    logits_check_policy: str,
    numeric_task_mode: str,
) -> dict[str, Any]:
    last = repeat_results[-1]
    return {
        "workload_id": plan["workload_id"],
        "status": "pass" if all_success(repeat_results) else "fail",
        "run_prepared_status": int(last["run_prepared_status"]),
        "repeat_runs": len(repeat_results),
        "planned_decode_steps": int(plan["decode_steps"]),
        "executed_decode_steps": (
            len(repeat_results) if decode_step_limit is not None else 0
        ),
        "decode_step_limit": decode_step_limit,
        "execution_mode": (
            "bounded_decode_steps"
            if decode_step_limit is not None
            else "repeat_submissions"
        ),
        "logits_check_policy": logits_check_policy,
        "numeric_task_mode": numeric_task_mode_summary(numeric_task_mode),
        "logits_check_summary": logits_check_summary(repeat_results),
        "repeat_results": repeat_results,
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
    return all(item["status"] == "pass" for item in repeat_results)


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
) -> list[str]:
    contracts = implemented_contracts(
        decode_step_limit,
        token_feedback_status=decode_feedback_contract_status(workload_results),
    )
    if numeric_task_mode == "unit_math":
        contracts.append("qwen_resource_backed_unit_numeric_task_mode")
    return contracts
