#!/usr/bin/env python3
"""Emit the PTO CUDA persistent-device Qwen serving lifecycle scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
TARGET_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}


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


def text_contains(path: str, needles: list[str]) -> bool:
    full_path = ROOT / path
    if not full_path.is_file():
        return False
    text = full_path.read_text(encoding="utf-8", errors="replace")
    return all(needle in text for needle in needles)


def serving_workload_contracts() -> list[dict[str, Any]]:
    payload = load_json(VIEWER_DATA / "serving_workloads.json")
    workloads = []
    for item in payload.get("serving_workloads", []):
        if item.get("id") not in TARGET_WORKLOAD_IDS:
            continue
        model_policy = item.get("model_policy", {})
        prompt_policy = item.get("prompt_policy", {})
        decode_policy = item.get("decode_policy", {})
        workloads.append(
            {
                "id": item.get("id"),
                "primary_model": model_policy.get("primary_model"),
                "target_prompt_tokens": prompt_policy.get("target_prompt_tokens"),
                "decode_tokens": decode_policy.get("decode_tokens"),
                "batch_sizes": decode_policy.get("batch_sizes", []),
            }
        )
    return workloads


def stage(
    *,
    stage_id: str,
    title: str,
    owner: str,
    required_for_full_serving: bool,
    status: str,
    evidence: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "title": title,
        "owner": owner,
        "required_for_full_serving": required_for_full_serving,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def build_scaffold() -> dict[str, Any]:
    persistent_abi_ready = text_contains(
        "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
        ["PtoCudaPersistentDagTask", "tensor_args", "scalar_args"],
    )
    persistent_codegen_ready = text_contains(
        "simpler_setup/cuda_callable_compiler.py",
        ["render_persistent_dag_source", "CudaPersistentTaskBodyFunction"],
    )
    stages = [
        stage(
            stage_id="serving_workload_contract",
            title="Shared Qwen serving workload contracts",
            owner="benchmark_viewer",
            required_for_full_serving=True,
            status="pass" if serving_workload_contracts() else "fail",
            evidence=(
                "docs/nvidia-backend/benchmark-viewer/data/"
                "serving_workloads.json"
            ),
            next_action="Keep MPK and VDCores serving policies in viewer data.",
        ),
        stage(
            stage_id="persistent_device_task_abi",
            title="Persistent-device task ABI",
            owner="pto_persistent_device",
            required_for_full_serving=True,
            status="pass" if persistent_abi_ready else "fail",
            evidence="src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
            next_action="Expose any additional tensor/scalar slots required by Qwen decode tasks.",
        ),
        stage(
            stage_id="persistent_dag_codegen",
            title="Persistent DAG task code generation",
            owner="pto_persistent_device",
            required_for_full_serving=True,
            status="pass" if persistent_codegen_ready else "fail",
            evidence="simpler_setup/cuda_callable_compiler.py",
            next_action="Add generated Qwen decode task bodies once kernels are selected.",
        ),
        stage(
            stage_id="qwen_tokenizer",
            title="Qwen tokenizer and prompt accounting",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="missing",
            evidence="none",
            next_action="Add repo-owned tokenizer or validated tokenizer adapter for Qwen prompts.",
        ),
        stage(
            stage_id="qwen_weight_loader",
            title="Qwen weight loader",
            owner="pto_serving_host",
            required_for_full_serving=True,
            status="missing",
            evidence="none",
            next_action="Load Qwen/Qwen3-8B tensors and bind them to persistent-device task args.",
        ),
        stage(
            stage_id="kv_cache_lifecycle",
            title="KV-cache allocation and token-position lifecycle",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status="missing",
            evidence="none",
            next_action="Represent paged or contiguous KV cache buffers in the persistent-device graph.",
        ),
        stage(
            stage_id="decode_loop_runner",
            title="Decode loop runner",
            owner="pto_serving_runtime",
            required_for_full_serving=True,
            status="missing",
            evidence="none",
            next_action="Run repeated decode steps, sampling, EOS policy, and output-token accounting.",
        ),
        stage(
            stage_id="viewer_result_import",
            title="Full-serving viewer result import",
            owner="paper_evaluation",
            required_for_full_serving=True,
            status="missing",
            evidence="docs/nvidia-backend/benchmark-viewer/data/results.json",
            next_action=(
                "Import Qwen/Qwen3-8B PTO rows with serving_coverage="
                "full_serving or full_serving_latency_caveat."
            ),
        ),
    ]
    missing = [
        item
        for item in stages
        if item["required_for_full_serving"] and item["status"] != "pass"
    ]
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_serving_scaffold",
        "status": "partial" if missing else "pass",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "runtime": "cuda/persistent_device",
        "serving_workloads": serving_workload_contracts(),
        "stages": stages,
        "missing_stage_ids": [item["id"] for item in missing],
        "next_action": (
            "Implement the missing PTO serving host/runtime stages, then "
            "import Qwen/Qwen3-8B full-serving rows."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_scaffold()
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
