from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403
from .pto_full_serving import validate_pto_full_serving_result


def validate_capture_imports(
    data: dict[str, Any],
    benchmark_ids: set[str],
    method_ids: set[str],
) -> None:
    hardware = require_dict(data, "hardware", "capture imports")
    for machine, record in hardware.items():
        if not isinstance(machine, str) or not machine:
            fail("capture imports hardware machine is empty")
        if not isinstance(record, dict):
            fail(f"capture imports hardware {machine} is not an object")
        for key in ("gpu", "compute_target"):
            require_string(record, key, f"capture imports hardware {machine}")

    records = require_list(data, "capture_imports", "capture imports")
    import_keys: set[tuple[str, int, int, tuple[int, int, int] | None]] = set()
    for record in records:
        if not isinstance(record, dict):
            fail("capture import rule is not an object")
        owner = f"capture import {record.get('baseline', '<missing>')}"
        baseline = require_string(record, "baseline", owner)
        validate_id(baseline, owner)
        benchmark_id = require_string(record, "benchmark_id", owner)
        method_id = require_string(record, "method_id", owner)
        if benchmark_id not in benchmark_ids:
            fail(f"{owner} references unknown benchmark_id: {benchmark_id}")
        if method_id not in method_ids:
            fail(f"{owner} references unknown method_id: {method_id}")
        n = record.get("n")
        task_count = record.get("task_count")
        for key in ("n", "task_count"):
            value = record.get(key)
            if not isinstance(value, int) or value <= 0:
                fail(f"{owner} has invalid {key}")
        tensor_tile = optional_tensor_tile(record, owner)
        import_key = (baseline, int(n), int(task_count), tensor_tile)
        if import_key in import_keys:
            fail(
                "duplicate capture import rule: "
                f"baseline={baseline}, n={n}, task_count={task_count}"
            )
        import_keys.add(import_key)
        inputs = require_dict(record, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)


def optional_tensor_tile(
    record: dict[str, Any],
    owner: str,
) -> tuple[int, int, int] | None:
    value = record.get("tensor_tile")
    if value is None:
        return None
    if not isinstance(value, dict):
        fail(f"{owner} tensor_tile is not an object")
    tile = []
    for key in ("rows", "cols", "inner"):
        field = value.get(key)
        if not isinstance(field, int) or field <= 0:
            fail(f"{owner} tensor_tile has invalid {key}")
        tile.append(field)
    return tuple(tile)


def validate_results(
    data: dict[str, Any],
    benchmark_ids: set[str],
    method_ids: set[str],
    root: Path,
) -> None:
    snapshot = require_dict(data, "snapshot", "results")
    require_string(snapshot, "commit", "results snapshot")
    if len(snapshot["commit"]) < 7:
        fail("results snapshot commit is too short")
    for key in ("full_capture", "compact_capture"):
        capture = require_dict(snapshot, key, "results snapshot")
        if not isinstance(capture.get("samples"), int) or capture["samples"] <= 0:
            fail(f"results snapshot {key} has invalid sample count")
        require_current_artifact_path(
            root,
            require_string(capture, "artifact_root", "results snapshot"),
            f"results snapshot {key}",
        )

    for key in ("headline_results", "selected_rows", "result_records"):
        require_list(data, key, "results")

    allowed_serving_coverage = {
        "full_serving",
        "full_serving_latency_caveat",
        "controlled_attention_tile_proxy",
        "diagnostic_microdecode",
        "diagnostic_qwen_descriptor_smoke",
        "diagnostic_resource_backed_qwen_dag",
        "diagnostic_unit_math",
        "native_bringup",
    }

    for record in data["result_records"]:
        if not isinstance(record, dict):
            fail("result record is not an object")
        owner = f"result {record.get('benchmark_id', '<missing>')}"
        benchmark_id = require_string(record, "benchmark_id", owner)
        method_id = require_string(record, "method_id", owner)
        if benchmark_id not in benchmark_ids:
            fail(f"{owner} references unknown benchmark_id: {benchmark_id}")
        if method_id not in method_ids:
            fail(f"{owner} references unknown method_id: {method_id}")
        require_string(record, "commit", owner)
        hardware = require_dict(record, "hardware", owner)
        for key in ("gpu", "machine", "compute_target", "driver", "cuda_toolkit"):
            require_string(hardware, key, owner)
        inputs = require_dict(record, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)
        statistic = require_dict(record, "statistic", owner)
        require_string(statistic, "kind", owner)
        if benchmark_id == "llm_serving_decode":
            coverage = require_string(statistic, "serving_coverage", owner)
            if coverage not in allowed_serving_coverage:
                fail(f"{owner} has invalid statistic.serving_coverage")
            if coverage != "full_serving" and "full-serving" in str(
                inputs.get("shape", "")
            ):
                fail(f"{owner} has proxy coverage but claims full-serving shape")
            if (
                method_id == "pto_persistent_device"
                and coverage == "full_serving"
                and "Qwen/Qwen3-8B" in str(inputs.get("shape", ""))
            ):
                validate_pto_full_serving_result(record, statistic, owner)
        sample_count = statistic.get("sample_count")
        if not isinstance(sample_count, int) or sample_count <= 0:
            fail(f"{owner} has invalid statistic.sample_count")
        for key in ("host_wall_ns", "device_wall_ns"):
            if not isinstance(statistic.get(key), int) or statistic[key] < 0:
                fail(f"{owner} has invalid statistic.{key}")
        if statistic["kind"] == "median_capture_group" and sample_count > 1:
            for prefix in ("host_wall", "device_wall"):
                for suffix in ("p50", "p90", "p99", "mean", "stdev", "min", "max"):
                    key = f"{prefix}_{suffix}_ns"
                    if not isinstance(statistic.get(key), int) or statistic[key] < 0:
                        fail(f"{owner} has invalid statistic.{key}")
                if statistic[f"{prefix}_min_ns"] > statistic[f"{prefix}_max_ns"]:
                    fail(f"{owner} has invalid {prefix} min/max statistic")
        raw_artifact = require_string(record, "raw_artifact", owner)
        require_current_artifact_path(root, raw_artifact, owner)
        if require_string(record, "correctness", owner) not in {
            "pass",
            "fail",
            "skipped",
            "not_applicable",
        }:
            fail(f"{owner} has invalid correctness: {record['correctness']}")
