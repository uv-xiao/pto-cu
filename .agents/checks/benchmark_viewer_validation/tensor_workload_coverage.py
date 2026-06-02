from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import check_evidence_refs


def _result_index(results: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    return {
        (
            record["benchmark_id"],
            record["method_id"],
            record["hardware"]["gpu"],
            record["inputs"]["shape"],
        )
        for record in results["result_records"]
    }


def _check_result_refs(
    refs: list[Any],
    owner: str,
    result_index: set[tuple[str, str, str, str]],
) -> None:
    for ref in refs:
        if not isinstance(ref, dict):
            fail(f"{owner} result ref is not an object")
        key = (
            require_string(ref, "benchmark_id", owner),
            require_string(ref, "method_id", owner),
            require_string(ref, "gpu", owner),
        )
        shape_contains = require_string(ref, "shape_contains", owner)
        if not any(
            result[:3] == key and shape_contains in result[3]
            for result in result_index
        ):
            fail(f"{owner} result ref is missing: {(*key, shape_contains)}")


def validate_tensor_workload_coverage(
    data: dict[str, Any],
    results: dict[str, Any],
    root: Path,
) -> None:
    metadata = require_dict(data, "metadata", "tensor workload coverage")
    for key in ("title", "status", "summary"):
        require_string(metadata, key, "tensor workload coverage metadata")
    groups = require_list(data, "coverage_groups", "tensor workload coverage")
    group_ids = check_unique_ids(groups, "tensor workload coverage group")
    required = {
        "pto_tensor_descriptor_paths",
        "library_and_generated_baselines",
        "tuned_pto_tensor_body_gap",
    }
    missing = required - group_ids
    if missing:
        fail(f"tensor workload coverage missing groups: {sorted(missing)}")
    results_seen = _result_index(results)
    for group in groups:
        owner = f"tensor workload coverage group {group['id']}"
        for key in ("title", "status", "summary"):
            require_string(group, key, owner)
        covered = require_list(group, "covered_cases", owner)
        if len(covered) < 5:
            fail(f"{owner} must list at least five covered cases")
        result_refs = group.get("result_refs", [])
        if group["status"] != "open" and not result_refs:
            fail(f"{owner} must include result refs")
        if not isinstance(result_refs, list):
            fail(f"{owner} result_refs is not a list")
        _check_result_refs(result_refs, owner, results_seen)
        require_list(group, "open_work", owner)
        check_evidence_refs(group, owner, root)
