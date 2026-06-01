#!/usr/bin/env python3
"""Import PTO Qwen resource-backed diagnostic execution into viewer results."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_RESULTS = VIEWER_DATA / "results.json"
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"
COVERAGE = "diagnostic_resource_backed_qwen_dag"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return result.stdout.strip() if result.returncode == 0 else "current-working"


def build_result_records(
    payload: dict[str, Any],
    *,
    raw_artifact: str,
    commit: str,
) -> list[dict[str, Any]]:
    execution = payload["resource_backed_execution"]
    device = execution["device"]
    records = []
    for workload in execution["workloads"]:
        counters = workload["scheduler_counters"]
        records.append(
            {
                "benchmark_id": "llm_serving_decode",
                "method_id": "pto_persistent_device",
                "hardware": {
                    "gpu": "A100",
                    "machine": "hina",
                    "compute_target": str(device.get("arch", "compute_80")),
                    "driver": "see raw artifact",
                    "cuda_toolkit": "see raw artifact",
                    "clock_policy": "not recorded in current snapshot",
                },
                "commit": commit,
                "inputs": {
                    "shape": (
                        "Qwen/Qwen3-8B resource-backed diagnostic "
                        f"{workload['workload_id']}, graph_tasks="
                        f"{workload['graph_task_count']}"
                    ),
                    "dtype": "float32 diagnostic task bodies with real buffers",
                    "repeat_policy": (
                        "single resource-backed run_prepared submission "
                        "inside one CUDA context"
                    ),
                },
                "statistic": {
                    "kind": "pto_qwen_resource_backed_execution",
                    "sample_count": 1,
                    "host_wall_ns": int(workload["timing_ns"]["host_wall"]),
                    "device_wall_ns": int(workload["timing_ns"]["device_wall"]),
                    "run_prepared_status": int(workload["run_prepared_status"]),
                    "completed_count": int(counters["completed_count"]),
                    "error_count": int(counters["error_count"]),
                    "scheduler_processed_count": int(
                        counters["scheduler_processed_count"],
                    ),
                    "task_count": int(workload["graph_task_count"]),
                    "serving_coverage": COVERAGE,
                    "workload_id": workload["workload_id"],
                    "context_policy": execution["context_policy"],
                },
                "raw_artifact": raw_artifact,
                "correctness": "pass" if workload["status"] == "pass" else "fail",
            }
        )
    return records


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
    updated = dict(results)
    merged = {result_key(item): item for item in results["result_records"]}
    for record in records:
        merged[result_key(record)] = record
    updated["result_records"] = list(merged.values())
    return updated


def ensure_matrix_ref(matrix: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "kind": "viewer_result",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "gpu": "A100",
        "shape_contains": "Qwen/Qwen3-8B resource-backed diagnostic",
        "serving_coverage": COVERAGE,
    }
    updated = dict(matrix)
    records = []
    for record in matrix["paper_evaluation_matrix"]:
        current = dict(record)
        if current["id"] == "llm_serving_paper_baselines":
            refs = current["current_evidence_refs"]
            if ref not in refs:
                refs.append(ref)
            for detail in current.get("missing_evidence_details", []):
                if detail.get("id") == "pto_full_serving_qwen3_8b":
                    detail["action"] = detail["action"].replace(
                        "diagnostic proxy, unit-math, and descriptor-smoke "
                        "viewer_result_imports are present.",
                        (
                            "diagnostic proxy, unit-math, descriptor-smoke, "
                            "and resource-backed execution viewer_result_imports "
                            "are present."
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
    payload = load_json(args.raw_json)
    raw_artifact = args.artifact_root or repo_relative(args.raw_json)
    records = build_result_records(
        payload,
        raw_artifact=raw_artifact,
        commit=args.commit,
    )
    write_json(args.results, merge_results(load_json(args.results), records))
    write_json(args.matrix, ensure_matrix_ref(load_json(args.matrix)))
    print(f"imported {raw_artifact}")


if __name__ == "__main__":
    main()
