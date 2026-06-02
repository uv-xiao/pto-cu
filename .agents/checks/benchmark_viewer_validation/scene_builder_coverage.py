from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import check_evidence_refs


def validate_scene_builder_coverage(data: dict[str, Any], root: Path) -> None:
    metadata = require_dict(data, "metadata", "scene builder coverage")
    for key in ("title", "status", "summary"):
        require_string(metadata, key, "scene builder coverage metadata")
    groups = require_list(data, "coverage_groups", "scene builder coverage")
    group_ids = check_unique_ids(groups, "scene builder coverage group")
    required = {
        "host_schedule_argument_builders",
        "persistent_graph_descriptor_builders",
        "real_data_scene_paths",
    }
    missing = required - group_ids
    if missing:
        fail(f"scene builder coverage missing groups: {sorted(missing)}")
    for group in groups:
        owner = f"scene builder coverage group {group['id']}"
        for key in ("title", "status", "summary"):
            require_string(group, key, owner)
        covered = require_list(group, "covered_builders", owner)
        if len(covered) < 3:
            fail(f"{owner} must list at least three covered builders")
        require_list(group, "open_work", owner)
        check_evidence_refs(group, owner, root)
