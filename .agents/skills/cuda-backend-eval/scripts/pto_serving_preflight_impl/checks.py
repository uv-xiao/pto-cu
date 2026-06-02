"""Reviewable PTO Qwen full-serving preflight checks."""

from __future__ import annotations

from typing import Any

from pto_serving_preflight_impl.constants import PAPER_WORKLOAD_IDS
from pto_serving_preflight_impl.io_helpers import text_contains


def status_if(condition: bool) -> str:
    return "pass" if condition else "fail"


def build_preflight_checks(
    *,
    serving_scaffold: dict[str, Any],
    proxy_rows: list[dict[str, Any]],
    qwen8b_missing_workloads: list[str],
    qwen8b_present_workloads: list[str],
    qwen8b_row_statuses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return (
        core_checks(proxy_rows)
        + qwen_input_checks(serving_scaffold)
        + qwen_weight_checks(serving_scaffold)
        + qwen_runtime_checks(serving_scaffold)
        + [
            {
                "id": "qwen3_8b_full_serving_rows_imported",
                "status": status_if(not qwen8b_missing_workloads),
                "evidence": "evaluations/nvidia/benchmark-viewer/data/results.json",
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
                "id": "qwen_model_loader_or_token_loop",
                "status": "fail",
                "evidence": "examples/cuda/persistent_qwen_serving_scaffold.py",
                "why": (
                    "PTO Qwen lifecycle stages are still missing: "
                    + ", ".join(serving_scaffold.get("missing_stage_ids", []))
                ),
            },
        ]
    )


def core_checks(proxy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": "persistent_device_task_descriptor_abi",
            "status": status_if(
                text_contains(
                    "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
                    ["PtoCudaPersistentDagTask", "tensor_args", "scalar_args"],
                )
            ),
            "evidence": "src/cuda/platform/include/host/pto_cuda_persistent_device_abi.h",
            "why": "Current persistent-device ABI carries DAG tasks plus generic tensor/scalar slots.",
        },
        {
            "id": "persistent_dag_source_codegen",
            "status": status_if(
                text_contains(
                    "simpler_setup/cuda_callable_compiler.py",
                    ["render_persistent_dag_source", "CudaPersistentTaskBodyFunction"],
                )
            ),
            "evidence": "simpler_setup/cuda_callable_compiler.py",
            "why": "Current compiler path can render persistent DAG task bodies.",
        },
        {
            "id": "pto_controlled_serving_proxy_imported",
            "status": status_if(bool(proxy_rows)),
            "evidence": "evaluations/nvidia/benchmark-viewer/data/results.json",
            "why": "Viewer contains the current PTO attention-tile serving-equivalent proxy row.",
        },
    ]


def qwen_input_checks(serving_scaffold: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        scaffold_check(
            "qwen_serving_lifecycle_scaffold",
            bool(serving_scaffold.get("status")),
            "examples/cuda/persistent_qwen_serving_scaffold.py",
            "Repo-owned scaffold declares the PTO Qwen full-serving lifecycle stages.",
        ),
        scaffold_check(
            "qwen_serving_lifecycle_plan",
            serving_scaffold.get("lifecycle_plan", {}).get("kind")
            == "pto_qwen_persistent_serving_lifecycle_plan",
            "examples/cuda/qwen_serving_lifecycle_plan.py",
            "Repo-owned plan maps Qwen3-8B serving policies to KV-cache capacity and persistent-device task roles.",
        ),
        scaffold_check(
            "qwen_prompt_accounting",
            serving_scaffold.get("prompt_accounting", {}).get("kind")
            == "pto_qwen_prompt_accounting",
            "examples/cuda/qwen_prompt_accounting.py",
            "Repo-owned prompt-accounting adapter records tokenizer availability and Qwen3-8B prompt counts.",
        ),
        scaffold_check(
            "qwen_runtime_input_binding",
            serving_scaffold.get("runtime_input_binding", {}).get("status")
            == "runtime_input_binding_plan_ready",
            "examples/cuda/qwen_runtime_input_binding.py",
            "Runtime input binding turns tokenizer outputs into padded input_ids, attention_mask, and output_ids descriptors.",
        ),
        scaffold_check(
            "qwen_cuda_token_buffer_binding",
            serving_scaffold.get("cuda_token_buffer_binding", {}).get("status")
            in {"token_buffer_binding_plan_ready", "cuda_token_buffer_binding_ready"},
            "examples/cuda/qwen_cuda_token_buffer_binding.py",
            "CUDA token-buffer binding maps token tensors into planned CUDA buffers and can run a live allocation/copy probe.",
        ),
        scaffold_check(
            "qwen_persistent_decode_args",
            serving_scaffold.get("persistent_decode_args", {}).get("status")
            in {"persistent_decode_args_plan_ready", "persistent_decode_args_ready"},
            "examples/cuda/qwen_persistent_decode_args.py",
            "Persistent decode args map token device pointers onto DAG task fields while preserving tensor_args for Qwen weights.",
        ),
        scaffold_check(
            "qwen_token_pointer_table_owner",
            serving_scaffold.get("token_pointer_table", {}).get("status")
            == "token_pointer_table_lifecycle_ready",
            "examples/cuda/qwen_token_pointer_table.py",
            "Token pointer-table lifecycle keeps Qwen token device pointers live while persistent decode args are materialized.",
        ),
    ]


def qwen_weight_checks(serving_scaffold: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        scaffold_check(
            "qwen_weight_inventory",
            serving_scaffold.get("weight_inventory", {}).get("kind")
            == "pto_qwen_weight_inventory",
            "examples/cuda/qwen_weight_inventory.py",
            "Safetensors inventory maps Qwen3-8B shards, weight groups, and expected shapes before runtime tensor binding.",
        ),
        scaffold_check(
            "qwen_safetensors_shard_plan",
            serving_scaffold.get("safetensors_shards", {}).get("kind")
            == "pto_qwen_safetensors_shard_status",
            "examples/cuda/qwen_safetensors_fetch.py",
            "Shard status records Qwen3-8B safetensors URLs, present/missing counts, and resumable fetch commands.",
        ),
        scaffold_check(
            "qwen_safetensors_shards_present",
            serving_scaffold.get("safetensors_shards", {}).get("status")
            == "ready_for_metadata_probe",
            "examples/cuda/qwen_safetensors_fetch.py",
            "All Qwen/Qwen3-8B safetensors shards must exist before metadata validation.",
        ),
        scaffold_check(
            "qwen_safetensors_metadata_probe",
            serving_scaffold.get("safetensors_metadata", {}).get("kind")
            == "pto_qwen_safetensors_metadata_probe",
            "examples/cuda/qwen_safetensors_metadata.py",
            "Safetensors metadata probe parses shard headers and compares them against the shape contract.",
        ),
        scaffold_check(
            "qwen_actual_safetensors_metadata",
            serving_scaffold.get("safetensors_metadata", {}).get("status")
            == "metadata_validated",
            "examples/cuda/qwen_safetensors_metadata.py",
            "Qwen/Qwen3-8B safetensors headers must match the expected shape/dtype contract.",
        ),
        scaffold_check(
            "qwen_cuda_weight_binding_plan",
            serving_scaffold.get("cuda_weight_binding", {}).get("status")
            == "binding_plan_ready",
            "examples/cuda/qwen_cuda_weight_binding.py",
            "Validated tensors must map to stable CUDA binding slots and persistent-device readonly weight roles.",
        ),
        scaffold_check(
            "qwen_persistent_weight_arg_manifest",
            serving_scaffold.get("persistent_weight_args", {}).get("status")
            == "persistent_weight_args_ready",
            "examples/cuda/qwen_persistent_weight_args.py",
            "Qwen weights must decompose into persistent DAG task tensor_args descriptors that fit the ABI.",
        ),
    ]


def qwen_runtime_checks(serving_scaffold: dict[str, Any]) -> list[dict[str, Any]]:
    decode_runner = serving_scaffold.get("decode_loop_runner", {})
    return [
        scaffold_check(
            "qwen_persistent_weight_materialization_plan",
            serving_scaffold.get("persistent_weight_materialization", {}).get("status")
            in {
                "persistent_weight_materialization_plan_ready",
                "persistent_weight_materialization_ready",
            },
            "examples/cuda/qwen_persistent_weight_materialization.py",
            "Persistent weight task descriptors must be materialized with resident device pointers.",
        ),
        scaffold_check(
            "qwen_resident_weight_table_owner",
            serving_scaffold.get("resident_weight_table", {}).get("status")
            == "resident_weight_table_lifecycle_ready",
            "examples/cuda/qwen_resident_weight_table.py",
            "Host side must own resident weight pointers through the whole decode-loop DAG submission lifetime.",
        ),
        scaffold_check(
            "qwen_kv_cache_binding",
            serving_scaffold.get("kv_cache_binding", {}).get("status")
            == "kv_cache_lifecycle_ready",
            "examples/cuda/qwen_kv_cache_binding.py",
            "KV-cache binding splits the planned cache into key/value device pointers and maps them to DAG c/d fields.",
        ),
        scaffold_check(
            "qwen_decode_loop_runner",
            decode_runner.get("status") == "decode_loop_runner_plan_ready"
            and "cuda_live_resource_bridge_contract"
            in decode_runner.get("implemented_contracts", []),
            "examples/cuda/qwen_decode_loop_runner.py",
            "Decode-loop runner orders token, KV-cache, and resident-weight owner lifetimes around persistent DAG submission.",
        ),
        scaffold_check(
            "qwen_persistent_task_bodies",
            serving_scaffold.get("persistent_task_bodies", {}).get("status")
            == "generated_task_bodies_ready",
            "examples/cuda/qwen_persistent_task_bodies.py",
            "Task-body source generation renders Qwen persistent-device callables through the persistent DAG source generator.",
        ),
    ]


def scaffold_check(
    check_id: str,
    condition: bool,
    evidence: str,
    why: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status_if(condition),
        "evidence": evidence,
        "why": why,
    }
