#!/usr/bin/env python3
"""Capture current PTO persistent-device full-serving readiness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_OUTPUT = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-preflight"
    / "pto-serving-preflight.json"
)
SERVING_SCAFFOLD = ROOT / "examples" / "cuda" / "persistent_qwen_serving_scaffold.py"


def fail(message: str) -> None:
    raise SystemExit(f"pto serving preflight failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def text_contains(path: str, needles: list[str]) -> bool:
    full_path = ROOT / path
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="replace")
    return all(needle in text for needle in needles)


def pto_serving_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in results.get("result_records", []):
        if not isinstance(record, dict):
            continue
        if record.get("benchmark_id") != "llm_serving_decode":
            continue
        if record.get("method_id") != "pto_persistent_device":
            continue
        rows.append(record)
    return rows


def serving_policy_summaries(serving_workloads: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for workload in serving_workloads.get("serving_workloads", []):
        if not isinstance(workload, dict):
            continue
        if workload.get("id") not in {"mpk_offline_decode", "vdcores_offline_decode"}:
            continue
        model_policy = workload.get("model_policy", {})
        prompt_policy = workload.get("prompt_policy", {})
        decode_policy = workload.get("decode_policy", {})
        summaries.append(
            {
                "id": workload.get("id", ""),
                "primary_model": model_policy.get("primary_model", ""),
                "target_prompt_tokens": prompt_policy.get("target_prompt_tokens"),
                "decode_tokens": decode_policy.get("decode_tokens"),
                "batch_sizes": decode_policy.get("batch_sizes", []),
            }
        )
    return summaries


def load_serving_scaffold() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "persistent_qwen_serving_scaffold",
        SERVING_SCAFFOLD,
    )
    if spec is None or spec.loader is None:
        fail(f"could not load {SERVING_SCAFFOLD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_scaffold()


def build_preflight() -> dict[str, Any]:
    serving_workloads = load_json(VIEWER_DATA / "serving_workloads.json")
    results = load_json(VIEWER_DATA / "results.json")
    serving_scaffold = load_serving_scaffold()
    pto_rows = pto_serving_rows(results)
    qwen8b_pto_rows = [
        row
        for row in pto_rows
        if "Qwen/Qwen3-8B" in str(row.get("inputs", {}).get("shape", ""))
    ]
    proxy_rows = [
        row
        for row in pto_rows
        if "attention tile proxy" in str(row.get("inputs", {}).get("shape", ""))
    ]

    checks = [
        {
            "id": "persistent_device_task_descriptor_abi",
            "status": "pass"
            if text_contains(
                "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
                ["PtoCudaPersistentDagTask", "tensor_args", "scalar_args"],
            )
            else "fail",
            "evidence": "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
            "why": "Current persistent-device ABI carries DAG tasks plus generic tensor/scalar slots.",
        },
        {
            "id": "persistent_dag_source_codegen",
            "status": "pass"
            if text_contains(
                "simpler_setup/cuda_callable_compiler.py",
                ["render_persistent_dag_source", "CudaPersistentTaskBodyFunction"],
            )
            else "fail",
            "evidence": "simpler_setup/cuda_callable_compiler.py",
            "why": "Current compiler path can render persistent DAG task bodies.",
        },
        {
            "id": "pto_controlled_serving_proxy_imported",
            "status": "pass" if proxy_rows else "fail",
            "evidence": "docs/nvidia-backend/benchmark-viewer/data/results.json",
            "why": "Viewer contains the current PTO attention-tile serving-equivalent proxy row.",
        },
        {
            "id": "qwen3_8b_full_serving_rows_imported",
            "status": "pass" if qwen8b_pto_rows else "fail",
            "evidence": "docs/nvidia-backend/benchmark-viewer/data/results.json",
            "why": "Full-serving readiness requires PTO rows whose shape names Qwen/Qwen3-8B.",
        },
        {
            "id": "qwen_serving_lifecycle_scaffold",
            "status": "pass" if serving_scaffold.get("status") else "fail",
            "evidence": "examples/cuda/persistent_qwen_serving_scaffold.py",
            "why": "Repo-owned scaffold declares the PTO Qwen full-serving lifecycle stages.",
        },
        {
            "id": "qwen_serving_lifecycle_plan",
            "status": "pass"
            if serving_scaffold.get("lifecycle_plan", {}).get("kind")
            == "pto_qwen_persistent_serving_lifecycle_plan"
            else "fail",
            "evidence": "examples/cuda/qwen_serving_lifecycle_plan.py",
            "why": (
                "Repo-owned plan maps Qwen3-8B serving policies to KV-cache "
                "capacity and persistent-device task roles."
            ),
        },
        {
            "id": "qwen_prompt_accounting",
            "status": "pass"
            if serving_scaffold.get("prompt_accounting", {}).get("kind")
            == "pto_qwen_prompt_accounting"
            else "fail",
            "evidence": "examples/cuda/qwen_prompt_accounting.py",
            "why": (
                "Repo-owned prompt-accounting adapter records tokenizer "
                "availability and Qwen3-8B prompt counts for the serving policies."
            ),
        },
        {
            "id": "qwen_model_loader_or_token_loop",
            "status": "fail",
            "evidence": "examples/cuda/persistent_qwen_serving_scaffold.py",
            "why": (
                "PTO Qwen lifecycle stages are still missing: "
                + ", ".join(serving_scaffold.get("missing_stage_ids", []))
            ),
        },
    ]
    blocking_gaps = [
        check["why"] for check in checks if check["status"] != "pass"
    ]
    return {
        "schema_version": 1,
        "kind": "pto_persistent_device_full_serving_preflight",
        "status": "partial" if blocking_gaps else "pass",
        "commit": git_commit(),
        "serving_workloads": serving_policy_summaries(serving_workloads),
        "serving_lifecycle": serving_scaffold,
        "pto_serving_rows": [
            {
                "shape": row.get("inputs", {}).get("shape", ""),
                "raw_artifact": row.get("raw_artifact", ""),
                "correctness": row.get("correctness", ""),
            }
            for row in pto_rows
        ],
        "checks": checks,
        "blocking_gaps": blocking_gaps,
        "next_action": (
            "Implement and import PTO persistent-device Qwen/Qwen3-8B "
            "full-serving rows for mpk_offline_decode and vdcores_offline_decode."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_preflight()
    write_json(args.output, payload)
    print(repo_relative(args.output))


if __name__ == "__main__":
    main()
