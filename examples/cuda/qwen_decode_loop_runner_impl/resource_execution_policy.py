"""Execution policy helpers for resource-backed Qwen DAG submissions."""

from __future__ import annotations

from typing import Any


def resource_backed_execution_count(
    *,
    plan: dict[str, Any],
    repeat_runs: int,
    decode_step_limit: int | None,
) -> int:
    if decode_step_limit is None:
        return max(1, int(repeat_runs))
    return max(1, min(int(plan["decode_steps"]), int(decode_step_limit)))


def decode_step_execution_summary(
    workload_results: list[dict[str, Any]],
    *,
    decode_step_limit: int | None,
) -> dict[str, Any]:
    if decode_step_limit is None:
        return {"status": "not_requested"}
    return {
        "status": "bounded_decode_steps_executed",
        "requested_decode_step_limit": int(decode_step_limit),
        "total_planned_decode_steps": sum(
            int(item.get("planned_decode_steps", 0)) for item in workload_results
        ),
        "total_executed_decode_steps": sum(
            int(item.get("executed_decode_steps", 0)) for item in workload_results
        ),
        "workloads": [
            {
                "workload_id": item["workload_id"],
                "planned_decode_steps": int(item.get("planned_decode_steps", 0)),
                "executed_decode_steps": int(item.get("executed_decode_steps", 0)),
            }
            for item in workload_results
        ],
    }


def implemented_contracts(
    decode_step_limit: int | None,
    *,
    token_feedback_status: str = "not_requested",
) -> list[str]:
    contracts = ["qwen_resource_backed_diagnostic_execution"]
    if decode_step_limit is not None:
        contracts.append("qwen_resource_backed_decode_step_execution")
    if token_feedback_status in {
        "diagnostic_token_feedback_applied",
        "device_token_feedback_observed",
    }:
        contracts.append("qwen_diagnostic_decode_token_feedback")
    if token_feedback_status == "device_token_feedback_observed":
        contracts.append("qwen_device_decode_token_feedback")
    return contracts


def logits_summary_stable(repeat_results: list[dict[str, Any]]) -> bool:
    if not repeat_results:
        return False
    first_key = logits_stability_key(repeat_results[0].get("logits_summary", {}))
    return all(
        logits_stability_key(item.get("logits_summary", {})) == first_key
        for item in repeat_results[1:]
    )


def logits_stability_key(summary: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    topk = summary.get("topk") or []
    return (
        summary.get("sample_checksum"),
        topk[0].get("token_id") if topk else None,
        summary.get("written_element_count"),
        summary.get("sampled_element_count"),
    )
