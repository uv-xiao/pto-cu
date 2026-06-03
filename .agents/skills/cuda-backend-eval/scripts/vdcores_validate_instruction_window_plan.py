#!/usr/bin/env python3
"""Validate VDCores shared-instruction window-plan artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_MODE = "vdcores_qwen3_8b_shared_instruction_window_plan"
EXPECTED_MODEL = "Qwen/Qwen3-8B"
EXPECTED_WORKLOAD = "vdcores_offline_decode"
EXPECTED_PREFERRED_PATH = "segmented_token_windowed_shared_instruction_schedule"


def _dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _positive_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and value > 0 else None


def _validate_windows(
    windows: list[Any],
    *,
    label: str,
    capacity: int,
    required: int,
    expected_count: int,
) -> list[str]:
    errors: list[str] = []
    if len(windows) != expected_count:
        errors.append(
            f"{label} window count mismatch: expected {expected_count}, "
            f"found {len(windows)}"
        )
    expected_start = 0
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            errors.append(f"{label} window {index} is not an object")
            continue
        if window.get("index") != index:
            errors.append(f"{label} window {index} has wrong index")
        start = window.get("instruction_start")
        end = window.get("instruction_end")
        count = window.get("instruction_count")
        if not all(isinstance(value, int) for value in (start, end, count)):
            errors.append(f"{label} window {index} has non-integer bounds")
            continue
        if start != expected_start:
            errors.append(
                f"{label} window {index} starts at {start}, expected "
                f"{expected_start}"
            )
        if end <= start:
            errors.append(f"{label} window {index} has non-positive range")
        if count != end - start:
            errors.append(f"{label} window {index} count does not match bounds")
        if count > capacity:
            errors.append(f"{label} window {index} exceeds shared capacity {capacity}")
        if window.get("capacity_ok") is not True:
            errors.append(f"{label} window {index} is not marked capacity_ok")
        expected_start = end
    if windows and expected_start != required:
        errors.append(
            f"{label} windows end at {expected_start}, expected {required}"
        )
    return errors


def validate_instruction_window_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if plan.get("mode") != EXPECTED_MODE:
        errors.append(f"mode must be {EXPECTED_MODE}")
    if plan.get("status") != "analysis_only":
        errors.append(
            "window plan must remain analysis_only until a runnable baseline exists"
        )
    if plan.get("model") != EXPECTED_MODEL:
        errors.append(f"model must be {EXPECTED_MODEL}")
    if plan.get("serving_workload_id") != EXPECTED_WORKLOAD:
        errors.append(f"serving_workload_id must be {EXPECTED_WORKLOAD}")

    capacity = _dict(plan, "shared_instruction_capacity")
    instructions_per_sm = _positive_int(capacity, "instructions_per_sm")
    num_sms = _positive_int(capacity, "num_sms")
    if instructions_per_sm is None:
        errors.append("missing shared_instruction_capacity.instructions_per_sm")
    if num_sms is None:
        errors.append("missing shared_instruction_capacity.num_sms")

    pressure = _dict(plan, "observed_instruction_pressure")
    max_cinsts = _positive_int(pressure, "max_cinsts_per_sm")
    max_minsts = _positive_int(pressure, "max_minsts_per_sm")
    if max_cinsts is None:
        errors.append("missing observed_instruction_pressure.max_cinsts_per_sm")
    if max_minsts is None:
        errors.append("missing observed_instruction_pressure.max_minsts_per_sm")

    lower_bound = _dict(plan, "minimum_window_lower_bound")
    compute_count = _positive_int(lower_bound, "compute_instruction_windows")
    memory_count = _positive_int(lower_bound, "memory_instruction_windows")
    worst_count = _positive_int(lower_bound, "worst_case_windows_per_sm")
    if compute_count is None:
        errors.append("missing minimum compute window count")
    if memory_count is None:
        errors.append("missing minimum memory window count")
    if worst_count is None:
        errors.append("missing minimum worst-case window count")
    if (
        compute_count is not None
        and memory_count is not None
        and worst_count is not None
        and worst_count != max(compute_count, memory_count)
    ):
        errors.append("worst-case window count does not match compute/memory counts")

    manifest = _dict(plan, "segmented_window_manifest")
    if manifest.get("manifest_kind") != "per_sm_uniform_lower_bound":
        errors.append("segmented manifest kind must be per_sm_uniform_lower_bound")
    if instructions_per_sm and max_cinsts and compute_count:
        errors.extend(
            _validate_windows(
                _list(manifest, "compute_instruction_windows"),
                label="compute",
                capacity=instructions_per_sm,
                required=max_cinsts,
                expected_count=compute_count,
            )
        )
    if instructions_per_sm and max_minsts and memory_count:
        errors.extend(
            _validate_windows(
                _list(manifest, "memory_instruction_windows"),
                label="memory",
                capacity=instructions_per_sm,
                required=max_minsts,
                expected_count=memory_count,
            )
        )
    if (
        instructions_per_sm is not None
        and manifest.get("max_compute_window_instruction_count")
        != instructions_per_sm
    ):
        errors.append("max compute window count must equal shared capacity")
    if (
        instructions_per_sm is not None
        and manifest.get("max_memory_window_instruction_count")
        != instructions_per_sm
    ):
        errors.append("max memory window count must equal shared capacity")

    runtime_change = _dict(plan, "required_runtime_change")
    if runtime_change.get("preferred_path") != EXPECTED_PREFERRED_PATH:
        errors.append(f"preferred_path must be {EXPECTED_PREFERRED_PATH}")
    for key in (
        "builder_requirements",
        "runtime_requirements",
        "pre_import_checks",
    ):
        if len(_list(runtime_change, key)) < 3:
            errors.append(f"required_runtime_change.{key} must list at least 3 items")
    return errors


def load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_instruction_window_plan(load_plan(args.path))
    if errors:
        raise SystemExit("\n".join(errors))
    print("vdcores instruction window plan validation passed")


if __name__ == "__main__":
    main()
