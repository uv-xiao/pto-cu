from __future__ import annotations

from .common import fail, require_dict, require_list, require_string


def validate_plan_history(data: dict) -> None:
    if data.get("schema_version") != 1:
        fail("plan history schema_version must be 1")
    require_string(data, "generated_at", "plan history")

    summary = require_dict(data, "summary", "plan history")
    for key in (
        "current_focus",
        "recent_pattern",
        "reflection",
        "test_strategy",
    ):
        require_string(summary, key, "plan history summary")
    if "benchmark model" not in summary["reflection"]:
        fail("plan history reflection must call out benchmark-model progress")
    if "row-by-row" not in summary["test_strategy"]:
        fail("plan history test strategy must discourage row-by-row tests")
    if "large" not in summary["test_strategy"]:
        fail("plan history test strategy must prefer large integrated tests")

    focus_windows = require_list(data, "work_focus", "plan history")
    latest = focus_windows[0]
    for key in ("feature_or_runtime", "tests_or_guardrails", "viewer_or_docs"):
        if not isinstance(latest.get(key), int) or latest[key] < 0:
            fail(f"plan history work_focus has invalid {key}")
    if latest["tests_or_guardrails"] <= latest["feature_or_runtime"]:
        fail("plan history must surface the current test-heavy work pattern")

    for record in require_list(data, "recent_slices", "plan history"):
        require_string(record, "commit", "plan history recent slice")
        require_string(record, "title", "plan history recent slice")
        require_string(record, "focus", "plan history recent slice")
        require_string(record, "reflection", "plan history recent slice")

    for record in require_list(data, "reflection_log", "plan history"):
        require_string(record, "date", "plan history reflection log")
        finding = require_string(record, "finding", "plan history reflection log")
        require_string(record, "decision", "plan history reflection log")
        if "too much time" not in finding:
            fail("plan history reflection log must call out excessive non-feature work")

    next_check = require_dict(
        data, "next_reflection_check", "plan history"
    )
    require_string(next_check, "cadence", "plan history next check")
    question = require_string(next_check, "question", "plan history next check")
    if "benchmark model" not in question:
        fail("plan history next check must ask about benchmark-model progress")
    require_string(
        next_check,
        "preferred_action_if_reporting_only",
        "plan history next check",
    )
