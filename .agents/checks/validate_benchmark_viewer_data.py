#!/usr/bin/env python3
"""Validate the CUDA benchmark viewer's review-facing data contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ID_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def fail(message: str) -> None:
    raise SystemExit(f"benchmark viewer data validation failed: {message}")


def load_json(root: Path, name: str) -> dict[str, Any]:
    path = (
        root / "docs" / "nvidia-backend" / "benchmark-viewer" / "data" / name
    )
    if not path.is_file():
        fail(f"missing data file: {path.relative_to(root)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path.relative_to(root)}: {exc}")


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} has empty or missing {key}")
    return value


def require_dict(record: dict[str, Any], key: str, owner: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict) or not value:
        fail(f"{owner} has empty or missing {key}")
    return value


def require_list(record: dict[str, Any], key: str, owner: str) -> list[Any]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        fail(f"{owner} has empty or missing {key}")
    return value


def validate_id(identifier: str, owner: str) -> None:
    if not ID_RE.fullmatch(identifier):
        fail(f"{owner} id is not stable snake_case: {identifier}")


def check_unique_ids(records: list[dict[str, Any]], owner: str) -> set[str]:
    ids: set[str] = set()
    for record in records:
        identifier = require_string(record, "id", owner)
        validate_id(identifier, owner)
        if identifier in ids:
            fail(f"duplicate {owner} id: {identifier}")
        ids.add(identifier)
    return ids


def check_evidence_refs(record: dict[str, Any], owner: str, root: Path) -> None:
    refs = require_list(record, "evidence_refs", owner)
    for ref in refs:
        if not isinstance(ref, dict):
            fail(f"{owner} evidence ref is not an object")
        relpath = require_string(ref, "path", owner)
        path = root / relpath
        if not path.is_file():
            fail(f"{owner} evidence path missing: {relpath}")
        text = path.read_text(encoding="utf-8", errors="replace")
        for symbol in require_list(ref, "symbols", owner):
            if not isinstance(symbol, str) or not symbol:
                fail(f"{owner} evidence symbol is empty")
            if symbol not in text:
                fail(f"{owner} missing evidence symbol {symbol} in {relpath}")


def validate_benchmarks(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "benchmarks", "benchmarks")
    benchmark_ids = check_unique_ids(records, "benchmark")
    for record in records:
        owner = f"benchmark {record['id']}"
        for key in ("title", "description", "math", "code"):
            require_string(record, key, owner)
        run = require_dict(record, "run", owner)
        require_string(run, "command", owner)
        inputs = require_dict(run, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)
        check_evidence_refs(record, owner, root)
    return benchmark_ids


def validate_methods(data: dict[str, Any], root: Path) -> set[str]:
    records = require_list(data, "methods", "methods")
    method_ids = check_unique_ids(records, "method")
    allowed_categories = {
        "pto_runtime",
        "vendor_baseline",
        "framework_baseline",
        "paper_baseline",
        "diagnostic_baseline",
    }
    for record in records:
        owner = f"method {record['id']}"
        for key in ("name", "runtime_flow", "lifecycle", "launch_model"):
            require_string(record, key, owner)
        category = require_string(record, "category", owner)
        if category not in allowed_categories:
            fail(f"{owner} has invalid category: {category}")
        check_evidence_refs(record, owner, root)
    return method_ids


def validate_paper_baselines(data: dict[str, Any]) -> set[str]:
    records = require_list(data, "paper_baselines", "paper_baselines")
    baseline_ids = check_unique_ids(records, "paper baseline")
    for record in records:
        owner = f"paper baseline {record['id']}"
        for key in ("name", "paper_role", "status", "next_action"):
            require_string(record, key, owner)
        source = require_dict(record, "source", owner)
        for key in ("upstream_url", "local_tmp_path", "commit"):
            require_string(source, key, owner)
        if len(source["commit"]) != 40:
            fail(f"{owner} source commit is not pinned: {source['commit']}")
        require_list(record, "paper_baselines_to_reproduce", owner)
    return baseline_ids


def validate_results(
    data: dict[str, Any], benchmark_ids: set[str], method_ids: set[str]
) -> None:
    snapshot = require_dict(data, "snapshot", "results")
    require_string(snapshot, "commit", "results snapshot")
    if len(snapshot["commit"]) < 7:
        fail("results snapshot commit is too short")
    for key in ("full_capture", "compact_capture"):
        capture = require_dict(snapshot, key, "results snapshot")
        if not isinstance(capture.get("samples"), int) or capture["samples"] <= 0:
            fail(f"results snapshot {key} has invalid sample count")
        require_string(capture, "artifact_root", "results snapshot")

    for key in ("headline_results", "selected_rows", "result_records"):
        require_list(data, key, "results")

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
        sample_count = statistic.get("sample_count")
        if not isinstance(sample_count, int) or sample_count <= 0:
            fail(f"{owner} has invalid statistic.sample_count")
        for key in ("host_wall_ns", "device_wall_ns"):
            if not isinstance(statistic.get(key), int) or statistic[key] < 0:
                fail(f"{owner} has invalid statistic.{key}")
        raw_artifact = require_string(record, "raw_artifact", owner)
        if not raw_artifact.startswith("tmp/"):
            fail(f"{owner} raw_artifact must be under tmp/: {raw_artifact}")
        if require_string(record, "correctness", owner) not in {
            "pass",
            "fail",
            "skipped",
            "not_applicable",
        }:
            fail(f"{owner} has invalid correctness: {record['correctness']}")


def validate_viewer_data(root: Path = ROOT) -> None:
    benchmarks = load_json(root, "benchmarks.json")
    methods = load_json(root, "methods.json")
    paper_baselines = load_json(root, "paper_baselines.json")
    results = load_json(root, "results.json")
    benchmark_ids = validate_benchmarks(benchmarks, root)
    method_ids = validate_methods(methods, root)
    validate_paper_baselines(paper_baselines)
    validate_results(results, benchmark_ids, method_ids)


def main() -> None:
    validate_viewer_data()
    print("benchmark viewer data validation passed")


if __name__ == "__main__":
    main()
