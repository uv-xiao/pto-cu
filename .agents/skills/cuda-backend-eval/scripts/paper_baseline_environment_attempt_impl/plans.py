"""Environment-plan lookup helpers."""

from __future__ import annotations

from typing import Any

from paper_baseline_environment_attempt_impl.errors import fail


def plan_for_baseline(plans: dict[str, Any], baseline_id: str) -> dict[str, Any]:
    records = plans.get("paper_baseline_environment_plans")
    if not isinstance(records, list):
        fail("paper_baseline_environment_plans is missing or not a list")
    for record in records:
        if isinstance(record, dict) and record.get("paper_baseline_id") == baseline_id:
            return record
    fail(f"missing environment plan for baseline: {baseline_id}")
