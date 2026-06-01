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


def implemented_contracts(decode_step_limit: int | None) -> list[str]:
    contracts = ["qwen_resource_backed_diagnostic_execution"]
    if decode_step_limit is not None:
        contracts.append("qwen_resource_backed_decode_step_execution")
    return contracts
