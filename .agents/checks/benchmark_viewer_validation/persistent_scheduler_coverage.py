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
