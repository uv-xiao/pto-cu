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


DEFAULT_RESULTS = VIEWER_DATA / "results.json"
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"
DEFAULT_RUN_READINESS = VIEWER_DATA / "paper_baseline_run_readiness.json"
DEFAULT_AUDIT = VIEWER_DATA / "paper_readiness_audit.json"


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
