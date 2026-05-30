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
    baselines: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            fail("capture import rule is not an object")
        owner = f"capture import {record.get('baseline', '<missing>')}"
        baseline = require_string(record, "baseline", owner)
        validate_id(baseline, owner)
        if baseline in baselines:
            fail(f"duplicate capture import baseline: {baseline}")
        baselines.add(baseline)
        benchmark_id = require_string(record, "benchmark_id", owner)
        method_id = require_string(record, "method_id", owner)
        if benchmark_id not in benchmark_ids:
            fail(f"{owner} references unknown benchmark_id: {benchmark_id}")
        if method_id not in method_ids:
            fail(f"{owner} references unknown method_id: {method_id}")
        for key in ("n", "task_count"):
            value = record.get(key)
            if not isinstance(value, int) or value <= 0:
                fail(f"{owner} has invalid {key}")
        inputs = require_dict(record, "inputs", owner)
        for key in ("shape", "dtype", "repeat_policy"):
            require_string(inputs, key, owner)


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
        require_string(statistic, "kind", owner)
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


def validate_paper_evaluation_matrix(
    data: dict[str, Any],
    benchmark_ids: set[str],
    method_ids: set[str],
    baseline_ids: set[str],
    results: dict[str, Any],
    root: Path,
) -> None:
    records = require_list(
        data, "paper_evaluation_matrix", "paper evaluation matrix"
    )
    matrix_ids = check_unique_ids(records, "paper evaluation matrix")
    required_claims = {
        "host_schedule_launch_overhead",
        "persistent_device_scheduler_overhead",
        "tensor_core_tile_baselines",
        "llm_serving_paper_baselines",
    }
    if not required_claims <= matrix_ids:
        missing = sorted(required_claims - matrix_ids)
        fail(f"missing paper evaluation matrix claims: {missing}")

    result_index = {
        (
            result["benchmark_id"],
            result["method_id"],
            result["hardware"]["gpu"],
        )
        for result in results["result_records"]
    }
    baseline_coverage: set[str] = set()
    method_coverage: set[str] = set()
    hardware_coverage: set[str] = set()
    allowed_status = {
        "planned_no_results",
        "partial_current_capture",
        "ready_for_paper_claim",
    }
    required_metrics = {"correctness", "raw_artifacts"}

    for record in records:
        owner = f"paper evaluation matrix {record['id']}"
        for key in ("title", "claim", "status", "promotion_gate"):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")

        workloads = require_list(record, "workload_ids", owner)
        methods = require_list(record, "method_ids", owner)
        baselines = record.get("paper_baseline_ids", [])
        if not isinstance(baselines, list):
            fail(f"{owner} paper_baseline_ids is not a list")
        hardware_targets = require_list(record, "hardware_targets", owner)
        metrics = set(require_list(record, "required_metrics", owner))
        evidence_refs = require_list(record, "current_evidence_refs", owner)
        require_list(record, "missing_evidence", owner)

        for workload_id in workloads:
            if workload_id not in benchmark_ids:
                fail(f"{owner} references unknown workload_id: {workload_id}")
        for method_id in methods:
            if method_id not in method_ids:
                fail(f"{owner} references unknown method_id: {method_id}")
            method_coverage.add(method_id)
        for baseline_id in baselines:
            if baseline_id not in baseline_ids:
                fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
            baseline_coverage.add(baseline_id)
        for hardware in hardware_targets:
            if not isinstance(hardware, str) or not hardware:
                fail(f"{owner} has invalid hardware target")
            hardware_coverage.add(hardware)
        for metric in metrics:
            if not isinstance(metric, str) or not metric:
                fail(f"{owner} has invalid required metric")
        if not required_metrics <= metrics:
            missing = sorted(required_metrics - metrics)
            fail(f"{owner} missing required metrics: {missing}")

        for ref in evidence_refs:
            if not isinstance(ref, dict):
                fail(f"{owner} current evidence ref is not an object")
            kind = require_string(ref, "kind", owner)
            if kind == "viewer_result":
                key = (
                    require_string(ref, "benchmark_id", owner),
                    require_string(ref, "method_id", owner),
                    require_string(ref, "gpu", owner),
                )
                if key not in result_index:
                    fail(f"{owner} viewer_result evidence is missing: {key}")
            elif kind in {
                "viewer_data",
                "stable_doc",
                "baseline_survey",
            }:
                path = require_string(ref, "path", owner)
                if not (root / path).is_file():
                    fail(f"{owner} evidence path missing: {path}")
            elif kind == "raw_artifact":
                path = require_string(ref, "path", owner)
                if not path.startswith("tmp/"):
                    fail(f"{owner} raw artifact evidence must be under tmp/: {path}")
            else:
                fail(f"{owner} has unknown evidence kind: {kind}")

    required_baselines = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    if not required_baselines <= baseline_coverage:
        missing = sorted(required_baselines - baseline_coverage)
        fail(f"paper evaluation matrix missing baseline coverage: {missing}")
    required_methods = {"pto_host_schedule", "pto_persistent_device"}
    if not required_methods <= method_coverage:
        missing = sorted(required_methods - method_coverage)
        fail(f"paper evaluation matrix missing PTO method coverage: {missing}")
    if not {"A100", "H200"} <= hardware_coverage:
        fail("paper evaluation matrix must cover A100 and H200")


def validate_viewer_data(root: Path = ROOT) -> None:
    benchmarks = load_json(root, "benchmarks.json")
    methods = load_json(root, "methods.json")
    paper_baselines = load_json(root, "paper_baselines.json")
    paper_evaluation_matrix = load_json(root, "paper_evaluation_matrix.json")
    capture_imports = load_json(root, "capture_imports.json")
    results = load_json(root, "results.json")
    benchmark_ids = validate_benchmarks(benchmarks, root)
    method_ids = validate_methods(methods, root)
    baseline_ids = validate_paper_baselines(paper_baselines)
    validate_capture_imports(capture_imports, benchmark_ids, method_ids)
    validate_results(results, benchmark_ids, method_ids)
    validate_paper_evaluation_matrix(
        paper_evaluation_matrix,
        benchmark_ids,
        method_ids,
        baseline_ids,
        results,
        root,
    )


def main() -> None:
    validate_viewer_data()
    print("benchmark viewer data validation passed")


if __name__ == "__main__":
    main()
