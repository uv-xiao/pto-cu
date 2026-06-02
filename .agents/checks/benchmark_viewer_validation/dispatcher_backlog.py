from __future__ import annotations

from typing import Any

from .common import *  # noqa: F403


def _status_gap_refs() -> set[str]:
    status = ROOT / "docs" / "nvidia-backend" / "status.md"
    text = status.read_text(encoding="utf-8")
    try:
        body = text.split("\n## Remaining Gaps\n", 1)[1].split("\n## ", 1)[0]
    except IndexError:
        fail("status.md has no Remaining Gaps section")
    refs: set[str] = set()
    for line in body.splitlines():
        if line.startswith("- [") and "](" in line and ")" in line:
            refs.add(line.split("](", 1)[1].split(")", 1)[0])
    if not refs:
        fail("status.md has no remaining-gap links")
    return refs


def validate_dispatcher_backlog(work_queue: dict[str, Any]) -> None:
    path = (
        ROOT
        / "docs"
        / "in_progress"
        / "nvidia_backend_paper_ready"
        / "evaluation_plan"
        / "dispatcher_backlog.md"
    )
    if not path.is_file():
        fail(f"missing {path.relative_to(ROOT)}")
    text = path.read_text(encoding="utf-8")
    for heading in (
        "## Completed First Pass",
        "## Active Backend Gaps",
        "## Active Paper Work Items",
        "## Promotion Rules",
    ):
        if heading not in text:
            fail(f"dispatcher backlog missing heading: {heading}")
    for stale in (
        "Add missing fields for method category",
        "Clone or inspect MPK and VDCores",
    ):
        if stale in text:
            fail(f"dispatcher backlog still contains stale first-pass task: {stale}")
    for relpath in _status_gap_refs():
        if relpath not in text:
            fail(f"dispatcher backlog missing status gap: {relpath}")
    work_items = require_list(work_queue, "work_items", "paper readiness work queue")
    expected_count = work_queue.get("summary", {}).get("total_work_items")
    if expected_count != len(work_items):
        fail("paper readiness work queue total does not match work_items")
    if f"`{expected_count}` active paper work items" not in text:
        fail("dispatcher backlog paper work item count is stale")
    for item in work_items:
        item_id = require_string(item, "id", "paper readiness work item")
        if item_id not in text:
            fail(f"dispatcher backlog missing work item: {item_id}")
        selectors = item.get("serving_command_plan_selectors", [])
        if not isinstance(selectors, list) or not all(
            isinstance(selector, str) for selector in selectors
        ):
            fail(f"{item_id} has invalid serving command selectors")
        for selector in selectors:
            if selector not in text:
                fail(
                    "dispatcher backlog missing serving command selector "
                    f"for {item_id}: {selector}"
                )
