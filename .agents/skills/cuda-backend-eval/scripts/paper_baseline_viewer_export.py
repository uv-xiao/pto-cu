#!/usr/bin/env python3
"""Export paper-baseline raw JSON into benchmark-viewer result records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"
DEFAULT_BENCHMARKS = VIEWER_DATA / "benchmarks.json"
DEFAULT_METHODS = VIEWER_DATA / "methods.json"
OPTIONAL_NUMERIC_METRICS = (
    "end_to_end_latency_ns",
    "time_to_first_token_ns",
    "inter_token_latency_ns",
    "throughput_tokens_per_s",
    "scheduler_overhead_ns",
    "queue_wait_ns",
    "ready_queue_depth",
    "task_count",
    "scheduler_count",
    "worker_count",
    "max_abs_error",
    "batch_size",
    "prompt_tokens",
    "decode_tokens",
)
OPTIONAL_STRUCTURED_METRICS = (
    "dispatch_trace",
    "resource_policy",
    "queue_pressure",
    "task_registry",
    "generated_kernel_metadata",
)


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline export failed: {message}")


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


def require_dict(record: dict[str, Any], key: str, owner: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict) or not value:
        fail(f"{owner} missing {key}")
    return value


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} missing {key}")
    return value


def require_number(record: dict[str, Any], key: str, owner: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        fail(f"{owner} has invalid {key}")
    return int(value)


def optional_number(record: dict[str, Any], key: str, owner: str) -> int | float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        fail(f"{owner} has invalid {key}")
    return value


def optional_structured_metric(
    record: dict[str, Any], key: str, owner: str
) -> dict[str, Any] | list[Any]:
    value = record.get(key)
    if not isinstance(value, (dict, list)) or not value:
        fail(f"{owner} has invalid {key}")
    return value


def index_by_id(data: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    records = data.get(key, [])
    if not isinstance(records, list):
        fail(f"{key} is not a list")
    return {record["id"]: record for record in records}


def result_record(
    raw: dict[str, Any],
    *,
    runs_by_id: dict[str, dict[str, Any]],
    benchmark_ids: set[str],
    method_ids: set[str],
    commit: str,
    raw_artifact: str,
) -> dict[str, Any]:
    owner = f"raw result {raw.get('paper_baseline_run_id', '<missing>')}"
    run_id = require_string(raw, "paper_baseline_run_id", owner)
    run = runs_by_id.get(run_id)
    if run is None:
        fail(f"{owner} references unknown paper_baseline_run_id")
    baseline_id = run["paper_baseline_id"]
    benchmark_id = require_string(raw, "benchmark_id", owner)
    if benchmark_id not in benchmark_ids:
        fail(f"{owner} references unknown benchmark_id: {benchmark_id}")
    if baseline_id not in method_ids:
        fail(f"{owner} baseline has no matching method_id: {baseline_id}")

    hardware = require_dict(raw, "hardware", owner)
    inputs = require_dict(raw, "inputs", owner)
    metrics = require_dict(raw, "metrics", owner)
    correctness = require_string(raw, "correctness", owner)
    if correctness not in {"pass", "fail", "skipped", "not_applicable"}:
        fail(f"{owner} has invalid correctness: {correctness}")

    statistic = {
        "kind": require_string(metrics, "kind", owner),
        "sample_count": require_number(metrics, "sample_count", owner),
        "host_wall_ns": int(optional_number(metrics, "host_wall_ns", owner))
        if "host_wall_ns" in metrics
        else int(optional_number(metrics, "end_to_end_latency_ns", owner))
        if "end_to_end_latency_ns" in metrics
        else 0,
        "device_wall_ns": int(optional_number(metrics, "device_wall_ns", owner))
        if "device_wall_ns" in metrics
        else 0,
    }
    for optional_key in OPTIONAL_NUMERIC_METRICS:
        if optional_key in metrics:
            statistic[optional_key] = optional_number(metrics, optional_key, owner)
    for optional_key in OPTIONAL_STRUCTURED_METRICS:
        if optional_key in metrics:
            statistic[optional_key] = optional_structured_metric(
                metrics, optional_key, owner
            )

    return {
        "benchmark_id": benchmark_id,
        "method_id": baseline_id,
        "hardware": {
            "gpu": require_string(hardware, "gpu", owner),
            "machine": require_string(hardware, "machine", owner),
            "compute_target": require_string(hardware, "compute_target", owner),
            "driver": str(hardware.get("driver", "see raw artifact")),
            "cuda_toolkit": str(hardware.get("cuda_toolkit", "see raw artifact")),
            "clock_policy": str(hardware.get("clock_policy", "not recorded")),
        },
        "commit": commit,
        "inputs": {
            "shape": require_string(inputs, "shape", owner),
            "dtype": require_string(inputs, "dtype", owner),
            "repeat_policy": require_string(inputs, "repeat_policy", owner),
        },
        "statistic": statistic,
        "raw_artifact": raw_artifact,
        "correctness": correctness,
    }


def export_paper_baseline_records(
    raw_payload: dict[str, Any],
    *,
    runs: dict[str, Any],
    benchmarks: dict[str, Any],
    methods: dict[str, Any],
    raw_artifact: str,
) -> list[dict[str, Any]]:
    runs_by_id = index_by_id(runs, "paper_baseline_runs")
    benchmark_ids = set(index_by_id(benchmarks, "benchmarks"))
    method_ids = set(index_by_id(methods, "methods"))
    metadata = raw_payload.get("metadata", {})
    commit = str(metadata.get("pto_commit", metadata.get("git_commit", "unknown")))
    raw_results = raw_payload.get("results", [])
    if not isinstance(raw_results, list) or not raw_results:
        fail("raw payload has no results")
    return [
        result_record(
            raw,
            runs_by_id=runs_by_id,
            benchmark_ids=benchmark_ids,
            method_ids=method_ids,
            commit=commit,
            raw_artifact=raw_artifact,
        )
        for raw in raw_results
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--benchmarks", type=Path, default=DEFAULT_BENCHMARKS)
    parser.add_argument("--methods", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--artifact-root")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_payload = load_json(args.raw_json)
    raw_artifact = args.artifact_root
    if raw_artifact is None:
        raw_artifact = repo_relative(args.raw_json.parent) + "/"
    records = export_paper_baseline_records(
        raw_payload,
        runs=load_json(args.runs),
        benchmarks=load_json(args.benchmarks),
        methods=load_json(args.methods),
        raw_artifact=raw_artifact,
    )
    if args.output:
        write_json(args.output, records)
    else:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
