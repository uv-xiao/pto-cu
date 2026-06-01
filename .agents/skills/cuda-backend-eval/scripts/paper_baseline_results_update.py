#!/usr/bin/env python3
"""Import paper-baseline raw JSON into viewer data and run status files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_baseline_viewer_export import (
    DEFAULT_BENCHMARKS,
    DEFAULT_METHODS,
    DEFAULT_RUNS,
    ROOT,
    VIEWER_DATA,
    export_paper_baseline_records,
    load_json,
    repo_relative,
    write_json,
)
from paper_readiness_audit import build_readiness_audit
from paper_readiness_audit import load_json as load_audit_json


DEFAULT_RESULTS = VIEWER_DATA / "results.json"
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"
DEFAULT_RUN_READINESS = VIEWER_DATA / "paper_baseline_run_readiness.json"
DEFAULT_ATTEMPTS = VIEWER_DATA / "paper_baseline_execution_attempts"
DEFAULT_AUDIT = VIEWER_DATA / "paper_readiness_audit.json"
REQUIRED_METRIC_KEYS = {
    "correctness": ("correctness",),
    "device_elapsed_time": ("device_wall_ns",),
    "scheduler_overhead": ("scheduler_overhead_ns",),
    "dispatch_trace": ("dispatch_trace",),
    "queue_pressure": ("queue_pressure",),
    "resource_policy": ("resource_policy",),
    "end_to_end_latency": ("end_to_end_latency_ns", "host_wall_ns"),
    "time_to_first_token": ("time_to_first_token_ns",),
    "inter_token_latency": ("inter_token_latency_ns",),
    "throughput": ("throughput_tokens_per_s", "throughput"),
    "model_and_prompt_shape": ("shape",),
    "batch_or_concurrency_policy": ("repeat_policy",),
    "tensor_shape": ("shape",),
    "raw_artifacts": ("raw_artifact",),
}


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline results update failed: {message}")


def require_raw_results(raw_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = raw_payload.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        fail("raw payload has no results")
    if not all(isinstance(record, dict) for record in raw_results):
        fail("raw payload results contain a non-object record")
    return raw_results


def imported_run_ids(raw_payload: dict[str, Any]) -> set[str]:
    run_ids: set[str] = set()
    for record in require_raw_results(raw_payload):
        run_id = record.get("paper_baseline_run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            fail("raw result missing paper_baseline_run_id")
        run_ids.add(run_id)
    return run_ids


def index_runs(runs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = runs.get("paper_baseline_runs")
    if not isinstance(records, list):
        fail("paper_baseline_runs payload has no paper_baseline_runs list")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            fail("paper_baseline_runs contains a non-object record")
        run_id = record.get("id")
        if isinstance(run_id, str):
            by_id[run_id] = record
    return by_id


def has_metric(raw: dict[str, Any], metric: str, raw_artifact: str) -> bool:
    if metric not in REQUIRED_METRIC_KEYS:
        return False
    metrics = raw.get("metrics")
    inputs = raw.get("inputs")
    if not isinstance(metrics, dict):
        return False
    if metric == "correctness":
        return raw.get("correctness") in {"pass", "fail", "skipped", "not_applicable"}
    if metric == "raw_artifacts":
        return isinstance(raw_artifact, str) and raw_artifact.startswith("tmp/")
    if metric in {"model_and_prompt_shape", "batch_or_concurrency_policy", "tensor_shape"}:
        if not isinstance(inputs, dict):
            return False
        return any(
            isinstance(inputs.get(key), str) and bool(inputs[key].strip())
            for key in REQUIRED_METRIC_KEYS[metric]
        )
    return any(key in metrics for key in REQUIRED_METRIC_KEYS[metric])


def validate_required_metrics(
    raw_payload: dict[str, Any],
    *,
    runs: dict[str, Any],
    raw_artifact: str,
) -> None:
    runs_by_id = index_runs(runs)
    for raw in require_raw_results(raw_payload):
        run_id = raw.get("paper_baseline_run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            fail("raw result missing paper_baseline_run_id")
        run = runs_by_id.get(run_id)
        if run is None:
            fail(f"raw result references unknown paper_baseline_run_id: {run_id}")
        required_metrics = run.get("required_metrics")
        if not isinstance(required_metrics, list) or not required_metrics:
            fail(f"paper baseline run {run_id} has no required_metrics")
        missing = [
            metric
            for metric in required_metrics
            if not isinstance(metric, str) or not has_metric(raw, metric, raw_artifact)
        ]
        if missing:
            fail(f"raw result {run_id} missing required metrics: {missing}")


def result_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    hardware = record.get("hardware", {})
    inputs = record.get("inputs", {})
    return (
        str(record.get("benchmark_id", "")),
        str(record.get("method_id", "")),
        str(hardware.get("gpu", "")),
        str(inputs.get("shape", "")),
        str(record.get("raw_artifact", "")),
    )


def merge_result_records(
    results: dict[str, Any],
    imported_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records = results.get("result_records")
    if not isinstance(records, list):
        fail("results payload has no result_records list")
    updated = dict(results)
    merged: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            fail("results payload contains a non-object result record")
        key = result_key(record)
        merged[key] = record
        order.append(key)
    for record in imported_records:
        key = result_key(record)
        if key not in merged:
            order.append(key)
        merged[key] = record
    updated["result_records"] = [merged[key] for key in order]
    return updated


def mark_runs_imported(runs: dict[str, Any], run_ids: set[str]) -> dict[str, Any]:
    records = runs.get("paper_baseline_runs")
    if not isinstance(records, list):
        fail("paper_baseline_runs payload has no paper_baseline_runs list")
    found: set[str] = set()
    updated = dict(runs)
    updated_records = []
    for record in records:
        if not isinstance(record, dict):
            fail("paper_baseline_runs contains a non-object record")
        current = dict(record)
        run_id = current.get("id")
        if run_id in run_ids:
            current["status"] = "imported_to_viewer"
            found.add(str(run_id))
        updated_records.append(current)
    missing = sorted(run_ids - found)
    if missing:
        fail(f"raw payload references unknown paper_baseline_run_id: {missing}")
    updated["paper_baseline_runs"] = updated_records
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("--artifact-root")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--benchmarks", type=Path, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--run-readiness", type=Path, default=DEFAULT_RUN_READINESS)
    parser.add_argument("--execution-attempts", type=Path, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--viewer-output", type=Path)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_payload = load_json(args.raw_json)
    raw_artifact = args.artifact_root
    if raw_artifact is None:
        raw_artifact = repo_relative(args.raw_json.parent) + "/"
    runs = load_json(args.runs)
    validate_required_metrics(
        raw_payload,
        runs=runs,
        raw_artifact=raw_artifact,
    )
    imported_records = export_paper_baseline_records(
        raw_payload,
        runs=runs,
        benchmarks=load_json(args.benchmarks),
        methods=load_json(args.methods),
        raw_artifact=raw_artifact,
    )
    if args.viewer_output:
        write_json(args.viewer_output, imported_records)

    updated_results = merge_result_records(
        load_json(args.results),
        imported_records,
    )
    updated_runs = mark_runs_imported(runs, imported_run_ids(raw_payload))
    write_json(args.results, updated_results)
    write_json(args.runs, updated_runs)

    audit = build_readiness_audit(
        matrix=load_json(args.matrix),
        runs=updated_runs,
        probes=load_json(args.probes),
        run_readiness=load_json(args.run_readiness),
        execution_attempts=load_audit_json(args.execution_attempts),
        results=updated_results,
    )
    write_json(args.audit_output, audit)
    print(
        "updated "
        f"{repo_relative(args.results)}, {repo_relative(args.runs)}, "
        f"and {repo_relative(args.audit_output)}"
    )


if __name__ == "__main__":
    main()
