#!/usr/bin/env python3
"""Import PTO Qwen full-serving raw results into benchmark-viewer rows."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_data_io import load_json as load_viewer_json
from viewer_data_io import write_json as write_viewer_json

VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_RESULTS = VIEWER_DATA / "results.json"
REQUIRED_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
REQUIRED_METRICS = {
    "sample_count",
    "host_wall_ns",
    "device_wall_ns",
    "end_to_end_latency_ns",
    "time_to_first_token_ns",
    "inter_token_latency_ns",
    "throughput_tokens_per_s",
}
CORRECTNESS_SCOPE = "full_qwen_numerical_correctness"
MODEL_ID = "Qwen/Qwen3-8B"
RUNTIME = "cuda/persistent_device"
SERVING_COVERAGE = "full_serving"
MIN_SAMPLE_COUNT = 3


def fail(message: str) -> None:
    raise SystemExit(f"pto qwen full-serving import failed: {message}")


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "current-working"

def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing raw JSON: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid raw JSON: {exc}")
    if not isinstance(payload, dict):
        fail("raw JSON root is not an object")
    return payload

def require_dict(record: dict[str, Any], key: str, owner: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        fail(f"{owner} missing {key}")
    return value


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} missing {key}")
    return value

def require_positive_int(record: dict[str, Any], key: str, owner: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{owner} has invalid {key}")
    return value


def require_nonnegative_int(record: dict[str, Any], key: str, owner: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{owner} has invalid {key}")
    return value

def require_positive_number(record: dict[str, Any], key: str, owner: str) -> int | float:
    value = record.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        fail(f"{owner} has invalid {key}")
    return value


def require_nonnegative_number(
    record: dict[str, Any],
    key: str,
    owner: str,
) -> int | float:
    value = record.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        fail(f"{owner} has invalid {key}")
    return value


def require_exact_int(
    record: dict[str, Any],
    key: str,
    expected: int,
    owner: str,
) -> int:
    value = require_nonnegative_int(record, key, owner)
    if value != expected:
        fail(f"{owner} has invalid {key}: expected {expected}, got {value}")
    return value


def correctness_details(raw: dict[str, Any], owner: str) -> dict[str, Any]:
    details = require_dict(raw, "correctness_details", owner)
    if require_string(details, "scope", owner) != CORRECTNESS_SCOPE:
        fail(f"{owner} has invalid correctness_details.scope")
    if require_string(details, "model_id", owner) != MODEL_ID:
        fail(f"{owner} has invalid correctness_details.model_id")
    if require_string(details, "status", owner) != "pass":
        fail(f"{owner} has invalid correctness_details.status")
    if details.get("token_match") is not True:
        fail(f"{owner} has invalid correctness_details.token_match")
    checked_token_count = require_positive_int(
        details,
        "checked_token_count",
        owner,
    )
    max_abs_error = require_nonnegative_number(details, "max_abs_error", owner)
    tolerance = require_positive_number(details, "tolerance", owner)
    if max_abs_error > tolerance:
        fail(f"{owner} exceeds correctness_details.tolerance")
    return {
        "scope": CORRECTNESS_SCOPE,
        "model_id": MODEL_ID,
        "status": "pass",
        "token_match": True,
        "checked_token_count": checked_token_count,
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
    }


def validate_full_serving_metrics(
    *,
    metrics: dict[str, Any],
    correctness: dict[str, Any],
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    sample_count: int,
    owner: str,
) -> None:
    if sample_count < MIN_SAMPLE_COUNT:
        fail(f"{owner} sample_count must be at least {MIN_SAMPLE_COUNT}")

    expected_input_tokens = batch_size * prompt_tokens
    expected_output_tokens = batch_size * decode_tokens
    checked_token_count = correctness["checked_token_count"]
    if checked_token_count < expected_output_tokens:
        fail(
            f"{owner} checked_token_count must cover generated tokens: "
            f"expected at least {expected_output_tokens}, got {checked_token_count}"
        )

    if "failed_requests" in metrics:
        failed_requests = require_nonnegative_int(metrics, "failed_requests", owner)
        if failed_requests != 0:
            fail(f"{owner} failed_requests must be zero")
    if "completed_requests" in metrics:
        require_exact_int(metrics, "completed_requests", batch_size, owner)
    if "total_input_tokens" in metrics:
        require_exact_int(metrics, "total_input_tokens", expected_input_tokens, owner)
    if "total_output_tokens" in metrics:
        require_exact_int(metrics, "total_output_tokens", expected_output_tokens, owner)


def raw_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        fail("raw payload has no results")
    if not all(isinstance(record, dict) for record in results):
        fail("raw payload results contain a non-object record")
    return results

def validate_workload_coverage(records: list[dict[str, Any]]) -> None:
    seen = {
        record.get("workload_id")
        for record in records
        if isinstance(record.get("workload_id"), str)
    }
    missing = sorted(REQUIRED_WORKLOAD_IDS - seen)
    if missing:
        fail(f"raw payload missing required workloads: {missing}")


def result_record(
    raw: dict[str, Any],
    *,
    raw_artifact: str,
    commit: str,
) -> dict[str, Any]:
    workload_id = require_string(raw, "workload_id", "raw result")
    if workload_id not in REQUIRED_WORKLOAD_IDS:
        fail(f"raw result has unsupported workload_id: {workload_id}")
    owner = f"raw result {workload_id}"
    if require_string(raw, "runtime", owner) != RUNTIME:
        fail(f"{owner} must declare runtime={RUNTIME}")
    if require_string(raw, "serving_coverage", owner) != SERVING_COVERAGE:
        fail(f"{owner} must declare serving_coverage={SERVING_COVERAGE}")
    if require_string(raw, "correctness", owner) != "pass":
        fail(f"{owner} must pass correctness before full-serving import")
    correctness = correctness_details(raw, owner)
    hardware = require_dict(raw, "hardware", owner)
    inputs = require_dict(raw, "inputs", owner)
    metrics = require_dict(raw, "metrics", owner)
    for metric in REQUIRED_METRICS:
        if metric not in metrics:
            fail(f"{owner} missing metric {metric}")

    batch_size = require_positive_int(inputs, "batch_size", owner)
    prompt_tokens = require_positive_int(inputs, "prompt_tokens", owner)
    decode_tokens = require_positive_int(inputs, "decode_tokens", owner)
    sample_count = require_positive_int(metrics, "sample_count", owner)
    device_wall_ns = require_nonnegative_int(metrics, "device_wall_ns", owner)
    validate_full_serving_metrics(
        metrics=metrics,
        correctness=correctness,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        sample_count=sample_count,
        owner=owner,
    )
    statistic = {
        "kind": "pto_qwen_full_serving_capture",
        "sample_count": sample_count,
        "host_wall_ns": require_positive_int(metrics, "host_wall_ns", owner),
        "device_wall_ns": device_wall_ns,
        "end_to_end_latency_ns": require_positive_int(
            metrics,
            "end_to_end_latency_ns",
            owner,
        ),
        "time_to_first_token_ns": require_positive_int(
            metrics,
            "time_to_first_token_ns",
            owner,
        ),
        "inter_token_latency_ns": require_positive_int(
            metrics,
            "inter_token_latency_ns",
            owner,
        ),
        "throughput_tokens_per_s": require_positive_number(
            metrics,
            "throughput_tokens_per_s",
            owner,
        ),
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "serving_coverage": "full_serving",
        "workload_id": workload_id,
        "correctness_scope": correctness["scope"],
        "checked_token_count": correctness["checked_token_count"],
        "max_abs_error": correctness["max_abs_error"],
        "correctness_tolerance": correctness["tolerance"],
    }
    for key in (
        "throughput",
        "total_token_throughput_tokens_per_s",
        "time_per_output_token_ns",
        "completed_requests",
        "max_concurrent_requests",
        "total_input_tokens",
        "total_output_tokens",
    ):
        if key in metrics:
            statistic[key] = require_positive_number(metrics, key, owner)
    if "failed_requests" in metrics:
        statistic["failed_requests"] = require_nonnegative_int(
            metrics,
            "failed_requests",
            owner,
        )

    shape = (
        f"{workload_id},Qwen/Qwen3-8B,batch={batch_size},"
        f"prompt_tokens={prompt_tokens},decode_tokens={decode_tokens},"
        "mode=pto_full_serving"
    )
    return {
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "hardware": {
            "gpu": require_string(hardware, "gpu", owner),
            "machine": require_string(hardware, "machine", owner),
            "compute_target": require_string(hardware, "compute_target", owner),
            "driver": str(hardware.get("driver", "see raw artifact")),
            "cuda_toolkit": str(hardware.get("cuda_toolkit", "see raw artifact")),
            "clock_policy": str(
                hardware.get("clock_policy", "not recorded in current snapshot"),
            ),
        },
        "commit": commit,
        "inputs": {
            "shape": str(inputs.get("shape", shape)),
            "dtype": require_string(inputs, "dtype", owner),
            "repeat_policy": require_string(inputs, "repeat_policy", owner),
        },
        "statistic": statistic,
        "raw_artifact": raw_artifact,
        "correctness": "pass",
        "correctness_details": correctness,
    }


def build_result_records(
    payload: dict[str, Any],
    *,
    raw_artifact: str,
    commit: str,
) -> list[dict[str, Any]]:
    if not raw_artifact.startswith("tmp/"):
        fail("raw artifact must be under tmp/")
    records = raw_results(payload)
    validate_workload_coverage(records)
    return [
        result_record(record, raw_artifact=raw_artifact, commit=commit)
        for record in records
    ]

def result_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        record["benchmark_id"],
        record["method_id"],
        record["hardware"]["gpu"],
        record["inputs"]["shape"],
        record["raw_artifact"],
    )


def merge_results(
    results: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    current_records = results.get("result_records")
    if not isinstance(current_records, list):
        fail("results payload has no result_records list")
    updated = dict(results)
    merged = {result_key(record): record for record in current_records}
    for record in records:
        merged[result_key(record)] = record
    updated["result_records"] = list(merged.values())
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("--artifact-root")
    parser.add_argument("--commit", default=current_commit())
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--viewer-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(args.raw_json)
    artifact_root = args.artifact_root or repo_relative(args.raw_json.parent) + "/"
    records = build_result_records(
        payload,
        raw_artifact=artifact_root,
        commit=args.commit,
    )
    if args.viewer_output:
        args.viewer_output.parent.mkdir(parents=True, exist_ok=True)
        args.viewer_output.write_text(
            json.dumps(records, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
    write_viewer_json(
        args.results,
        merge_results(load_viewer_json(args.results), records),
    )
    print(f"imported PTO Qwen full-serving rows from {artifact_root}")


if __name__ == "__main__":
    main()
