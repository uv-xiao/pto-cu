from __future__ import annotations

from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403
from .generated_builders import load_goal_progress_builder


def remaining_gap_refs_from_status() -> set[str]:
    status = ROOT / "docs" / "nvidia-backend" / "status.md"
    text = status.read_text(encoding="utf-8")
    try:
        body = text.split("\n## Remaining Gaps\n", 1)[1].split("\n## ", 1)[0]
    except IndexError:
        fail("status.md has no Remaining Gaps section")
    refs: set[str] = set()
    for line in body.splitlines():
        if not line.startswith("- [") or "](" not in line or ")" not in line:
            continue
        relpath = line.split("](", 1)[1].split(")", 1)[0]
        refs.add(f"docs/nvidia-backend/{relpath}")
    if not refs:
        fail("status.md has no remaining-gap links")
    return refs


def validate_goal_progress(
    goal_progress: dict[str, Any],
    *,
    audit: dict[str, Any],
    work_queue: dict[str, Any],
    matrix: dict[str, Any],
    baselines: dict[str, Any],
) -> None:
    if goal_progress.get("schema_version") != 1:
        fail("goal progress schema_version must be 1")
    if goal_progress.get("overall_status") not in {"complete", "in_progress"}:
        fail("goal progress overall_status is invalid")
    summary = goal_progress.get("summary")
    if not isinstance(summary, dict):
        fail("goal progress has no summary object")
    criteria = goal_progress.get("acceptance_criteria")
    if not isinstance(criteria, list) or not criteria:
        fail("goal progress has no acceptance_criteria list")
    if summary.get("criteria_total") != len(criteria):
        fail("goal progress criteria_total does not match criteria")
    criteria_ids: set[str] = set()
    statuses: dict[str, int] = {}
    for criterion in criteria:
        if not isinstance(criterion, dict):
            fail("goal progress criterion is not an object")
        owner = f"goal progress criterion {criterion.get('id', '<missing>')}"
        criterion_id = require_string(criterion, "id", owner)
        validate_id(criterion_id, owner)
        if criterion_id in criteria_ids:
            fail(f"duplicate goal progress criterion id: {criterion_id}")
        criteria_ids.add(criterion_id)
        status = require_string(criterion, "status", owner)
        if status not in {"met", "in_progress"}:
            fail(f"{owner} has invalid status: {status}")
        statuses[status] = statuses.get(status, 0) + 1
        for key in ("title", "summary"):
            require_string(criterion, key, owner)
        for key in ("evidence_refs", "verification", "gaps"):
            if not isinstance(criterion.get(key), list):
                fail(f"{owner} {key} is not a list")
        if not criterion["evidence_refs"]:
            fail(f"{owner} has no evidence_refs")
        if not criterion["verification"]:
            fail(f"{owner} has no verification")
        if status == "met" and criterion["gaps"]:
            fail(f"{owner} is met but still has gaps")
    required_ids = {
        "benchmark_viewer",
        "nvidia_examples",
        "stable_docs_evidence",
        "changelog_reports",
        "remote_evaluation",
        "paper_evaluation_plan",
        "backend_implementation_closure",
        "paper_grade_results",
        "dispatcher_log",
    }
    if criteria_ids != required_ids:
        fail(f"goal progress criteria ids are stale: {sorted(criteria_ids)}")
    if summary.get("criteria_met") != statuses.get("met", 0):
        fail("goal progress criteria_met does not match criteria")
    if summary.get("criteria_in_progress") != statuses.get("in_progress", 0):
        fail("goal progress criteria_in_progress does not match criteria")
    by_id = {criterion["id"]: criterion for criterion in criteria}
    backend_closure = by_id["backend_implementation_closure"]
    if backend_closure["status"] != "in_progress":
        fail("backend implementation closure must remain in_progress")
    expected_gap_refs = {
        "docs/nvidia-backend/status.md",
        *remaining_gap_refs_from_status(),
    }
    if set(backend_closure["evidence_refs"]) != expected_gap_refs:
        fail("backend implementation closure refs do not match status.md")
    paper_results = by_id["paper_grade_results"]
    if paper_results.get("paper_readiness_status") != audit.get("overall_status"):
        fail("goal progress paper readiness status does not match audit")
    queue_total = work_queue.get("summary", {}).get("total_work_items")
    if paper_results.get("blocking_work_items") != queue_total:
        fail("goal progress blocking work item count does not match queue")
    if audit.get("overall_status") != "paper_ready":
        if paper_results["status"] != "in_progress":
            fail("goal progress must keep paper results in progress")
        if not paper_results["gaps"]:
            fail("goal progress paper result gaps are empty")
    generated = load_goal_progress_builder()(
        audit=audit,
        work_queue=work_queue,
        matrix=matrix,
        baselines=baselines,
    )
    if goal_progress != generated:
        fail("goal progress is stale; regenerate goal_progress.json")
