"""Resource-backed Qwen workload and logits-check policy helpers."""

from __future__ import annotations

from typing import Any


LOGITS_CHECK_POLICIES = ("every_step", "final_step")


def normalize_logits_check_policy(policy: str) -> str:
    if policy not in LOGITS_CHECK_POLICIES:
        raise ValueError(f"unknown logits check policy: {policy}")
    return policy


def select_workload_plans(
    plans: list[dict[str, Any]],
    workload_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not workload_ids:
        return plans
    requested = set(workload_ids)
    return [plan for plan in plans if plan["workload_id"] in requested]


def should_check_logits(
    *,
    policy: str,
    repeat_index: int,
    execution_count: int,
) -> bool:
    if policy == "every_step":
        return True
    if policy == "final_step":
        return repeat_index == execution_count - 1
    raise ValueError(f"unknown logits check policy: {policy}")


def unchecked_logits_summary(
    *,
    policy: str,
    repeat_index: int,
    execution_count: int,
) -> dict[str, Any]:
    return {
        "coverage": "not_checked",
        "reason": "deferred_by_logits_check_policy",
        "logits_check_policy": policy,
        "repeat_index": int(repeat_index),
        "is_final_step": repeat_index == execution_count - 1,
        "logits_buffer_elements": 0,
        "written_element_count": 0,
        "sampled_element_count": 0,
        "diagnostic_reference": {
            "status": "not_checked",
            "checked_element_count": 0,
            "max_abs_error": 0.0,
        },
    }


def logits_check_summary(repeat_results: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [
        item
        for item in repeat_results
        if item.get("logits_summary", {}).get("coverage") != "not_checked"
    ]
    return {
        "checked_step_count": len(checked),
        "deferred_step_count": len(repeat_results) - len(checked),
        "checked_repeat_indices": [int(item["repeat_index"]) for item in checked],
    }


def logits_summary_stable_for_checked_steps(
    repeat_results: list[dict[str, Any]],
    stability_key,
) -> bool:
    checked = [
        item
        for item in repeat_results
        if item.get("logits_summary", {}).get("coverage") != "not_checked"
    ]
    if not checked:
        return False
    first_key = stability_key(checked[0].get("logits_summary", {}))
    return all(
        stability_key(item.get("logits_summary", {})) == first_key
        for item in checked[1:]
    )
