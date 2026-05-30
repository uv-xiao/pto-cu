#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in
# compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text.
# -----------------------------------------------------------------------------
"""Export CUDA benchmark captures into benchmark-viewer result records."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAPPING = (
    ROOT
    / "docs"
    / "nvidia-backend"
    / "benchmark-viewer"
    / "data"
    / "capture_imports.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def group_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("machine"),
            row.get("baseline"),
            int(row.get("n", 0)),
            int(row.get("task_count", 1)),
            int(row.get("worker_blocks_per_task", 1)),
        )
        grouped[key].append(row)
    return grouped


def matching_rows(
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]],
    rule: dict[str, Any],
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    matches = []
    for key, rows in grouped.items():
        _machine, baseline, n, task_count, _worker_blocks_per_task = key
        if baseline != rule["baseline"]:
            continue
        if "n" in rule and n != int(rule["n"]):
            continue
        if "task_count" in rule and task_count != int(rule["task_count"]):
            continue
        matches.append((key, rows))
    return sorted(
        matches,
        key=lambda item: (str(item[0][0]), item[0][2], item[0][3]),
    )


def median_int(rows: list[dict[str, Any]], field: str) -> int:
    values = [int(row[field]) for row in rows]
    return int(statistics.median(values))


def record_for_group(
    *,
    key: tuple[Any, ...],
    rows: list[dict[str, Any]],
    rule: dict[str, Any],
    hardware_by_machine: dict[str, dict[str, str]],
    commit: str,
    raw_artifact: str,
) -> dict[str, Any]:
    machine = str(key[0])
    hardware = hardware_by_machine.get(machine, {})
    first = rows[0]
    return {
        "benchmark_id": rule["benchmark_id"],
        "method_id": rule["method_id"],
        "hardware": {
            "gpu": hardware.get("gpu", machine),
            "machine": machine,
            "compute_target": hardware.get(
                "compute_target", str(first.get("ptx_arch", "unknown"))
            ),
            "driver": str(first.get("driver", "see raw artifact")),
            "cuda_toolkit": str(first.get("cuda_toolkit", "see raw artifact")),
            "clock_policy": str(
                first.get("clock_policy", "not recorded in current snapshot")
            ),
        },
        "commit": commit,
        "inputs": dict(rule["inputs"]),
        "statistic": {
            "kind": "median_capture_group",
            "sample_count": len(rows),
            "host_wall_ns": median_int(rows, "host_wall_ns"),
            "device_wall_ns": median_int(rows, "device_wall_ns"),
        },
        "raw_artifact": raw_artifact,
        "correctness": "pass"
        if all(row.get("status") == "pass" for row in rows)
        else "fail",
    }


def export_result_records(
    capture: dict[str, Any],
    mapping: dict[str, Any],
    raw_artifact: str,
) -> list[dict[str, Any]]:
    metadata = capture.get("metadata", {})
    commit = str(metadata.get("git_commit", "unknown"))
    hardware_by_machine = mapping.get("hardware", {})
    grouped = group_rows(capture.get("results", []))
    machine_rank = {
        machine: index for index, machine in enumerate(hardware_by_machine)
    }
    ranked_records: list[tuple[int, int, dict[str, Any]]] = []
    for rule_index, rule in enumerate(mapping.get("capture_imports", [])):
        for key, rows in matching_rows(grouped, rule):
            machine = str(key[0])
            record = (
                record_for_group(
                    key=key,
                    rows=rows,
                    rule=rule,
                    hardware_by_machine=hardware_by_machine,
                    commit=commit,
                    raw_artifact=raw_artifact,
                )
            )
            ranked_records.append(
                (machine_rank.get(machine, 999), rule_index, record)
            )
    return [record for _machine_rank, _rule_index, record in sorted(ranked_records)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_json", type=Path)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING,
        help="Viewer import mapping JSON.",
    )
    parser.add_argument(
        "--artifact-root",
        help="Repo-relative raw artifact path to store in each result record.",
    )
    parser.add_argument("--output", type=Path, help="Write result records here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = load_json(args.capture_json)
    mapping = load_json(args.mapping)
    raw_artifact = args.artifact_root
    if raw_artifact is None:
        raw_artifact = repo_relative(args.capture_json.parent) + "/"
    records = export_result_records(capture, mapping, raw_artifact)
    if args.output:
        write_json(args.output, records)
    else:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
