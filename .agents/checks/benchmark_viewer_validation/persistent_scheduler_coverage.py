from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import check_evidence_refs


def validate_persistent_scheduler_coverage(
    data: dict[str, Any],
    root: Path,
) -> None:
    metadata = require_dict(data, "metadata", "persistent scheduler coverage")
    for key in ("title", "status", "summary"):
        require_string(metadata, key, "persistent scheduler coverage metadata")
    groups = require_list(
        data,
        "coverage_groups",
        "persistent scheduler coverage",
    )
    group_ids = check_unique_ids(groups, "persistent scheduler coverage group")
    required = {
        "launch_and_lifecycle",
        "graph_shape_matrix",
        "scheduler_negative_matrix",
    }
    missing = required - group_ids
    if missing:
        fail(f"persistent scheduler coverage missing groups: {sorted(missing)}")
    for group in groups:
        owner = f"persistent scheduler coverage group {group['id']}"
        for key in ("title", "status", "summary"):
            require_string(group, key, owner)
        covered = require_list(group, "covered_cases", owner)
        if len(covered) < 5:
            fail(f"{owner} must list at least five covered cases")
        require_list(group, "open_work", owner)
        check_evidence_refs(group, owner, root)
    check_persistent_scheduler_gap_docs(root)


def check_persistent_scheduler_gap_docs(root: Path) -> None:
    gap_root = (
        root
        / "docs"
        / "nvidia-backend"
        / "status"
        / "remaining-gaps"
        / "persistent-scheduler-generalization"
    )
    coverage_path = (
        root
        / "docs"
        / "nvidia-backend"
        / "status"
        / "persistent-scheduler-coverage.md"
    )
    coverage_text = coverage_path.read_text(encoding="utf-8")
    stale_phrases = (
        "additional scheduler-negative cases beyond the current labeled",
        "and additional scheduler-negative coverage",
        "broader scheduler error taxonomy",
    )
    split_gap_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(gap_root.glob("*.md"))
    )
    for phrase in stale_phrases:
        if phrase in split_gap_text or phrase in coverage_text:
            fail(
                "persistent scheduler coverage reintroduced a standalone "
                f"negative-coverage blocker: {phrase}"
            )
    required_phrases = (
        "normal PTO graph breadth",
        "current scheduler-negative taxonomy is covered",
        "malformed normal-graph lowering cases",
    )
    combined = f"{split_gap_text}\n{coverage_text}"
    for phrase in required_phrases:
        if phrase not in combined:
            fail(
                "persistent scheduler gap docs must keep current taxonomy "
                f"covered and normal-graph breadth open: {phrase}"
            )
