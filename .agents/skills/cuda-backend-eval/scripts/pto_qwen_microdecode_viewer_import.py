#!/usr/bin/env python3
"""Import PTO Qwen proxy microdecode live evidence into viewer results."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return "current-working"
    return result.stdout.strip()


def median_int(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def build_result_record(
    payload: dict[str, Any],
    *,
    raw_artifact: str,
    commit: str,
) -> dict[str, Any]:
    observations = payload["decode_loop_observations"]
    host_times = [int(item["timing_ns"]["host_wall"]) for item in observations]
    device_times = [int(item["timing_ns"]["device_wall"]) for item in observations]
    summary = payload["decode_loop_summary"]
    return {
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "hardware": {
            "gpu": "A100",
            "machine": "hina",
            "compute_target": "compute_80",
            "driver": "see raw artifact",
            "cuda_toolkit": "see raw artifact",
            "clock_policy": "not recorded in current snapshot",
        },
        "commit": commit,
        "inputs": {
            "shape": "Qwen/Qwen3-8B controlled proxy microdecode loop, repeat_runs=3",
            "dtype": "float32 controlled proxy arithmetic",
            "repeat_policy": "single prepared callable reused across three run_prepared submissions",
        },
        "statistic": {
            "kind": "pto_qwen_proxy_microdecode_loop",
            "sample_count": int(summary["repeat_runs"]),
            "host_wall_ns": median_int(host_times),
            "device_wall_ns": median_int(device_times),
            "repeat_runs": int(summary["repeat_runs"]),
            "completed_count": int(summary["total_completed_count"]),
            "error_count": int(summary["total_error_count"]),
            "scheduler_processed_count": int(
                summary["total_scheduler_processed_count"]
            ),
            "max_abs_error": float(payload["max_abs_error"]),
            "task_count": int(payload["dag"]["task_count"]),
            "serving_coverage": "diagnostic_microdecode",
        },
        "raw_artifact": raw_artifact,
        "correctness": "pass" if payload["status"] == "pass" else "fail",
    }


def result_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        record["benchmark_id"],
        record["method_id"],
        record["hardware"]["gpu"],
        record["inputs"]["shape"],
        record["raw_artifact"],
    )


def merge_result(results: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(results)
    merged = {result_key(item): item for item in results["result_records"]}
    merged[result_key(record)] = record
    updated["result_records"] = list(merged.values())
    return updated


def ensure_matrix_ref(matrix: dict[str, Any]) -> dict[str, Any]:
    updated = dict(matrix)
    records = []
    ref = {
        "kind": "viewer_result",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "gpu": "A100",
        "shape_contains": "Qwen/Qwen3-8B controlled proxy microdecode loop",
        "serving_coverage": "diagnostic_microdecode",
    }
    for record in matrix["paper_evaluation_matrix"]:
        current = dict(record)
        if current["id"] == "llm_serving_paper_baselines":
            refs = current["current_evidence_refs"]
            if ref not in refs:
                refs.append(ref)
            for detail in current.get("missing_evidence_details", []):
                if detail.get("id") == "pto_full_serving_qwen3_8b":
                    detail["action"] = detail["action"].replace(
                        "viewer_result_import remain required",
                        (
                            "full-serving viewer_result_import remains required; "
                            "diagnostic proxy viewer_result_import is present"
                        ),
                    )
        records.append(current)
    updated["paper_evaluation_matrix"] = records
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--artifact-root")
    parser.add_argument("--commit", default=current_commit())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.raw_json.read_text(encoding="utf-8"))
    raw_artifact = args.artifact_root or repo_relative(args.raw_json)
    record = build_result_record(
        payload,
        raw_artifact=raw_artifact,
        commit=args.commit,
    )
    write_viewer_json(args.results, merge_result(load_viewer_json(args.results), record))
    write_viewer_json(
        args.matrix,
        ensure_matrix_ref(load_viewer_json(args.matrix)),
    )
    print(f"imported {raw_artifact}")


if __name__ == "__main__":
    main()
