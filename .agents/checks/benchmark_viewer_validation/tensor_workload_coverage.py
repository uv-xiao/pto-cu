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
    method_ids: set[str],
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
    validate_model_shape_targets(data, method_ids, root)
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


def validate_model_shape_targets(
    data: dict[str, Any],
    method_ids: set[str],
    root: Path,
) -> None:
    targets = require_list(
        data,
        "model_shape_targets",
        "tensor workload coverage",
    )
    if len(targets) < 2:
        fail("tensor workload coverage must include at least two model shape targets")
    required_methods = {
        "pto_persistent_device",
        "cublas_sgemm_graph",
        "cutlass",
        "triton",
        "thunderkittens",
    }
    for target in targets:
        owner = f"tensor workload model shape target {target.get('id')}"
        validate_id(require_string(target, "id", owner), owner)
        for key in ("title", "status", "model_mapping", "run_command"):
            require_string(target, key, owner)
        tile = require_dict(target, "tensor_tile", owner)
        rows = require_positive_int(tile, "rows", owner)
        cols = require_positive_int(tile, "cols", owner)
        inner = require_positive_int(tile, "inner", owner)
        if rows % 16 != 0 or cols % 16 != 0 or inner % 8 != 0:
            fail(f"{owner} is not compatible with WMMA tensor-core constraints")
        command = target["run_command"]
        for flag, value in (
            ("--tensor-rows", rows),
            ("--tensor-cols", cols),
            ("--tensor-inner", inner),
        ):
            if f"{flag} {value}" not in command:
                fail(f"{owner} command missing {flag} {value}")
        methods = set(require_list(target, "required_methods", owner))
        if methods != required_methods:
            fail(f"{owner} required methods mismatch: {sorted(methods)}")
        missing_methods = methods - method_ids
        if missing_methods:
            fail(
                f"{owner} references unknown viewer methods: "
                f"{sorted(missing_methods)}"
            )
        check_evidence_refs(target, owner, root)


def require_positive_int(record: dict[str, Any], key: str, owner: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or value <= 0:
        fail(f"{owner} has invalid positive integer {key}")
    return value
