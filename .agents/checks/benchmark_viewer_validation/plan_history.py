from __future__ import annotations

import subprocess

from .common import fail, require_dict, require_list, require_string


PLAN_HISTORY_PATH = "evaluations/nvidia/benchmark-viewer/data/plan_history.json"


def current_commit_updates_plan_history() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    changed_paths = [
        path.strip()
        for path in result.stdout.splitlines()
        if path.strip()
    ]
    return PLAN_HISTORY_PATH in changed_paths


def current_or_maintenance_parent_short_commits() -> set[str]:
    commits: set[str] = set()
    revisions = ["HEAD"]
    if current_commit_updates_plan_history():
        revisions.append("HEAD^")
    for revision in revisions:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short=8", revision],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        commits.add(result.stdout.strip())
    return commits


def validate_plan_history(
    data: dict,
    allowed_latest_commits: set[str] | None = None,
) -> None:
    if data.get("schema_version") != 1:
        fail("plan history schema_version must be 1")
    require_string(data, "generated_at", "plan history")
    latest_reviewed_commit = require_string(
        data, "latest_reviewed_commit", "plan history"
    )
    allowed = allowed_latest_commits
    if allowed is None:
        allowed = current_or_maintenance_parent_short_commits()
    if allowed and latest_reviewed_commit not in allowed:
        fail(
            "plan history latest_reviewed_commit must match the current "
            "checkout or a plan-history maintenance parent"
        )

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
    recent_slices = require_list(data, "recent_slices", "plan history")
    if len(recent_slices) > 8:
        fail("plan history recent_slices must stay brief")
    for record in recent_slices:
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
    action = require_string(
        next_check,
        "next_benchmark_model_action",
        "plan history next check",
    )
    if "Qwen" not in action and "benchmark" not in action:
        fail("plan history next action must name benchmark-model work")
