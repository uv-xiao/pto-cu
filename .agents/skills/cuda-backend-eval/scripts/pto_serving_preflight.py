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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_data_io import load_json as load_viewer_json

VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_OUTPUT = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-preflight"
    / "pto-serving-preflight.json"
)
SERVING_SCAFFOLD = ROOT / "examples" / "cuda" / "persistent_qwen_serving_scaffold.py"
PAPER_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
FULL_SERVING_METRIC_FIELDS = {
    "end_to_end_latency_ns",
    "inter_token_latency_ns",
    "time_to_first_token_ns",
    "throughput_tokens_per_s",
    "batch_size",
    "decode_tokens",
}


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


def row_workload_id(row: dict[str, Any]) -> str:
    statistic = row.get("statistic", {})
    workload_id = statistic.get("workload_id")
    if isinstance(workload_id, str) and workload_id:
        return workload_id
    shape = str(row.get("inputs", {}).get("shape", ""))
    for candidate in PAPER_WORKLOAD_IDS:
        if candidate in shape:
            return candidate
    return ""


def full_serving_qwen_row_status(row: dict[str, Any]) -> dict[str, Any]:
    statistic = row.get("statistic", {})
    shape = str(row.get("inputs", {}).get("shape", ""))
    workload_id = row_workload_id(row)
    missing = []
    if row.get("benchmark_id") != "llm_serving_decode":
        missing.append("benchmark_id=llm_serving_decode")
    if row.get("method_id") != "pto_persistent_device":
        missing.append("method_id=pto_persistent_device")
    if "Qwen/Qwen3-8B" not in shape:
        missing.append("inputs.shape contains Qwen/Qwen3-8B")
    if statistic.get("serving_coverage") != "full_serving":
        missing.append("statistic.serving_coverage=full_serving")
    if row.get("correctness") != "pass":
        missing.append("correctness=pass")
    if workload_id not in PAPER_WORKLOAD_IDS:
        missing.append("workload_id is mpk_offline_decode or vdcores_offline_decode")
    for key in sorted(FULL_SERVING_METRIC_FIELDS):
        value = statistic.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            missing.append(f"statistic.{key}>0")
    if not row.get("raw_artifact"):
        missing.append("raw_artifact")
    return {
        "status": "pass" if not missing else "fail",
        "workload_id": workload_id,
        "shape": shape,
        "raw_artifact": row.get("raw_artifact", ""),
        "correctness": row.get("correctness", ""),
        "serving_coverage": statistic.get("serving_coverage", ""),
        "missing_requirements": missing,
    }


def full_serving_qwen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if full_serving_qwen_row_status(row)["status"] == "pass"
    ]


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
    results = load_viewer_json(VIEWER_DATA / "results.json")
    serving_scaffold = load_serving_scaffold()
    pto_rows = pto_serving_rows(results)
    qwen8b_pto_rows = full_serving_qwen_rows(pto_rows)
    qwen8b_row_statuses = [
        full_serving_qwen_row_status(row)
        for row in pto_rows
        if "Qwen/Qwen3-8B" in str(row.get("inputs", {}).get("shape", ""))
    ]
    qwen8b_present_workloads = sorted(
        {row_workload_id(row) for row in qwen8b_pto_rows}
    )
    qwen8b_missing_workloads = sorted(
        PAPER_WORKLOAD_IDS - set(qwen8b_present_workloads)
    )
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
            "status": "pass" if not qwen8b_missing_workloads else "fail",
            "evidence": "docs/nvidia-backend/benchmark-viewer/data/results.json",
            "why": (
                "Full-serving readiness requires PTO Qwen/Qwen3-8B rows for "
                "mpk_offline_decode and vdcores_offline_decode with correctness "
                "and paper latency/throughput metrics."
            ),
            "required_workload_ids": sorted(PAPER_WORKLOAD_IDS),
            "present_workload_ids": qwen8b_present_workloads,
            "missing_workload_ids": qwen8b_missing_workloads,
            "row_statuses": qwen8b_row_statuses,
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
            "id": "qwen_runtime_input_binding",
            "status": "pass"
            if serving_scaffold.get("runtime_input_binding", {}).get("status")
            == "runtime_input_binding_plan_ready"
            else "fail",
            "evidence": "examples/cuda/qwen_runtime_input_binding.py",
            "why": (
                "Repo-owned runtime input binding turns tokenizer outputs "
                "into padded input_ids, attention_mask, and output_ids "
                "descriptors for persistent decode-loop submission."
            ),
        },
        {
            "id": "qwen_cuda_token_buffer_binding",
            "status": "pass"
            if serving_scaffold.get("cuda_token_buffer_binding", {}).get(
                "status"
            )
            in {
                "token_buffer_binding_plan_ready",
                "cuda_token_buffer_binding_ready",
            }
            else "fail",
            "evidence": "examples/cuda/qwen_cuda_token_buffer_binding.py",
            "why": (
                "Repo-owned CUDA token-buffer binding maps input_ids, "
                "attention_mask, and output_ids into planned CUDA buffers "
                "and can run a live allocation/copy verification probe."
            ),
        },
        {
            "id": "qwen_persistent_decode_args",
            "status": "pass"
            if serving_scaffold.get("persistent_decode_args", {}).get(
                "status"
            )
            in {
                "persistent_decode_args_plan_ready",
                "persistent_decode_args_ready",
            }
            else "fail",
            "evidence": "examples/cuda/qwen_persistent_decode_args.py",
            "why": (
                "Repo-owned persistent decode argument binding maps token "
                "device pointers onto the PtoCudaPersistentDagTask a/b/out "
                "fields while preserving tensor_args for Qwen weights."
            ),
        },
        {
            "id": "qwen_token_pointer_table_owner",
            "status": "pass"
            if serving_scaffold.get("token_pointer_table", {}).get("status")
            == "token_pointer_table_lifecycle_ready"
            else "fail",
            "evidence": "examples/cuda/qwen_token_pointer_table.py",
            "why": (
                "Repo-owned token pointer-table lifecycle keeps Qwen token "
                "device pointers live while persistent decode task arguments "
                "are materialized."
            ),
        },
        {
            "id": "qwen_weight_inventory",
            "status": "pass"
            if serving_scaffold.get("weight_inventory", {}).get("kind")
            == "pto_qwen_weight_inventory"
            else "fail",
            "evidence": "examples/cuda/qwen_weight_inventory.py",
            "why": (
                "Repo-owned safetensors inventory maps Qwen3-8B shards, "
                "weight groups, and expected shapes before runtime tensor "
                "binding."
            ),
        },
        {
            "id": "qwen_safetensors_shard_plan",
            "status": "pass"
            if serving_scaffold.get("safetensors_shards", {}).get("kind")
            == "pto_qwen_safetensors_shard_status"
            else "fail",
            "evidence": "examples/cuda/qwen_safetensors_fetch.py",
            "why": (
                "Repo-owned shard status records Qwen3-8B safetensors URLs, "
                "target paths, present/missing counts, and resumable fetch "
                "commands before any large download is attempted."
            ),
        },
        {
            "id": "qwen_safetensors_shards_present",
            "status": "pass"
            if serving_scaffold.get("safetensors_shards", {}).get("status")
            == "ready_for_metadata_probe"
            else "fail",
            "evidence": "examples/cuda/qwen_safetensors_fetch.py",
            "why": (
                "All Qwen/Qwen3-8B safetensors shards must exist locally "
                "before the metadata probe can validate actual tensor headers."
            ),
        },
        {
            "id": "qwen_safetensors_metadata_probe",
            "status": "pass"
            if serving_scaffold.get("safetensors_metadata", {}).get("kind")
            == "pto_qwen_safetensors_metadata_probe"
            else "fail",
            "evidence": "examples/cuda/qwen_safetensors_metadata.py",
            "why": (
                "Repo-owned safetensors metadata probe can parse shard "
                "headers and compare them against the expected shape contract."
            ),
        },
        {
            "id": "qwen_actual_safetensors_metadata",
            "status": "pass"
            if serving_scaffold.get("safetensors_metadata", {}).get("status")
            == "metadata_validated"
            else "fail",
            "evidence": "examples/cuda/qwen_safetensors_metadata.py",
            "why": (
                "Qwen/Qwen3-8B safetensors shards must be present locally and "
                "their actual header metadata must match the expected "
                "shape/dtype contract."
            ),
        },
        {
            "id": "qwen_cuda_weight_binding_plan",
            "status": "pass"
            if serving_scaffold.get("cuda_weight_binding", {}).get("status")
            == "binding_plan_ready"
            else "fail",
            "evidence": "examples/cuda/qwen_cuda_weight_binding.py",
            "why": (
                "Validated safetensors tensors must be mapped to stable "
                "CUDA binding slots, file byte ranges, binding groups, and "
                "persistent-device readonly weight argument roles before "
                "runtime device residency can be completed."
            ),
        },
        {
            "id": "qwen_persistent_weight_arg_manifest",
            "status": "pass"
            if serving_scaffold.get("persistent_weight_args", {}).get("status")
            == "persistent_weight_args_ready"
            else "fail",
            "evidence": "examples/cuda/qwen_persistent_weight_args.py",
            "why": (
                "Qwen weights must be decomposed into persistent DAG task "
                "tensor_args descriptors that fit the current four-pointer "
                "PtoCudaPersistentDagTask ABI before runtime pointer "
                "materialization can be implemented."
            ),
        },
        {
            "id": "qwen_persistent_weight_materialization_plan",
            "status": "pass"
            if serving_scaffold.get("persistent_weight_materialization", {}).get(
                "status"
            )
            in {
                "persistent_weight_materialization_plan_ready",
                "persistent_weight_materialization_ready",
            }
            else "fail",
            "evidence": "examples/cuda/qwen_persistent_weight_materialization.py",
            "why": (
                "Persistent weight task descriptors must be materialized with "
                "resident device pointers through the same ctypes layout used "
                "by persistent DAG submission."
            ),
        },
        {
            "id": "qwen_resident_weight_table_owner",
            "status": "pass"
            if serving_scaffold.get("resident_weight_table", {}).get("status")
            == "resident_weight_table_lifecycle_ready"
            else "fail",
            "evidence": "examples/cuda/qwen_resident_weight_table.py",
            "why": (
                "The host side must own resident weight pointers for the "
                "whole decode-loop DAG submission lifetime, then free them "
                "after submission completes."
            ),
        },
        {
            "id": "qwen_kv_cache_binding",
            "status": "pass"
            if serving_scaffold.get("kv_cache_binding", {}).get("status")
            == "kv_cache_lifecycle_ready"
            else "fail",
            "evidence": "examples/cuda/qwen_kv_cache_binding.py",
            "why": (
                "Repo-owned KV-cache binding splits the planned cache into "
                "key/value device pointers and maps them to persistent DAG "
                "c/d fields without consuming token or weight pointer slots."
            ),
        },
        {
            "id": "qwen_decode_loop_runner",
            "status": "pass"
            if serving_scaffold.get("decode_loop_runner", {}).get("status")
            == "decode_loop_runner_plan_ready"
            and "cuda_live_resource_bridge_contract"
            in serving_scaffold.get("decode_loop_runner", {}).get(
                "implemented_contracts",
                [],
            )
            else "fail",
            "evidence": "examples/cuda/qwen_decode_loop_runner.py",
            "why": (
                "Repo-owned decode-loop runner integration orders token, "
                "KV-cache, and resident-weight owner lifetimes around "
                "persistent DAG submission and records the diagnostic "
                "cuda_live bridge into the repeated microdecode runner."
            ),
        },
        {
            "id": "qwen_persistent_task_bodies",
            "status": "pass"
            if serving_scaffold.get("persistent_task_bodies", {}).get("status")
            == "generated_task_bodies_ready"
            else "fail",
            "evidence": "examples/cuda/qwen_persistent_task_bodies.py",
            "why": (
                "Repo-owned task-body source generation renders Qwen "
                "persistent-device callables through the existing persistent "
                "DAG source generator and records token, mutable KV-cache, "
                "and weight tensor-arg field consumption."
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
