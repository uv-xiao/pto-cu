"""Record identity helpers for paper serving command plans."""

from __future__ import annotations

from typing import Any

from paper_serving_command_plan_impl.errors import fail


def selected_model(workload: dict[str, Any], tier: str) -> str:
    key = f"{tier}_model"
    model = workload["model_policy"].get(key)
    if not model:
        fail(f"{workload['id']} has no model tier {tier!r}")
    return str(model)


def plan_id(run_id: str, policy_id: str, batch_size: int) -> str:
    return f"{run_id}:{policy_id}:batch{batch_size}"
