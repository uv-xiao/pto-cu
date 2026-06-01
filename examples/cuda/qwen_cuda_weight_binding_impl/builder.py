"""Top-level Qwen CUDA weight-binding artifact builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_COPY_CHUNK_BYTES,
    DEFAULT_HOST_RUNTIME,
    DEFAULT_INDEX,
    DEFAULT_SHARD_DIR,
    MODEL_ID,
    MODEL_REVISION,
    repo_relative,
)
from .cuda_runtime import run_cuda_copy_probe
from .full_residency import run_cuda_full_residency_probe
from .safetensors import build_bindings


def build_weight_binding(
    *,
    index_json: Path = DEFAULT_INDEX,
    weight_inventory_json: Path | None = None,
    metadata_json: Path | None = None,
    shard_dir: Path = DEFAULT_SHARD_DIR,
    no_cuda_probe: bool = False,
    device: int = 0,
    host_runtime: Path = DEFAULT_HOST_RUNTIME,
    cuda_probe_mode: str = "bounded",
    max_probe_tensor_bytes: int = 16 * 1024,
    max_probe_total_bytes: int = 256 * 1024,
    max_probe_tensors: int = 16,
    copy_chunk_bytes: int = DEFAULT_COPY_CHUNK_BYTES,
    verify_tensors: int = 8,
    verify_bytes: int = 4096,
) -> dict[str, Any]:
    summary, bindings = build_bindings(
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
        metadata_json=metadata_json,
        shard_dir=shard_dir,
    )
    cuda_probe = (
        {
            "mode": (
                "full_residency"
                if cuda_probe_mode == "full"
                else "bounded_copy"
            ),
            "status": "skipped",
            "reason": "disabled_by_no_cuda_probe",
        }
        if no_cuda_probe
        else run_cuda_full_residency_probe(
            bindings=bindings,
            device=device,
            host_runtime=host_runtime,
            chunk_bytes=copy_chunk_bytes,
            verify_tensors=verify_tensors,
            verify_bytes=verify_bytes,
        )
        if cuda_probe_mode == "full"
        else run_cuda_copy_probe(
            bindings=bindings,
            device=device,
            host_runtime=host_runtime,
            max_tensor_bytes=max_probe_tensor_bytes,
            max_total_bytes=max_probe_total_bytes,
            max_tensors=max_probe_tensors,
        )
    )
    implemented_contracts = [
        "safetensors_tensor_data_offsets",
        "persistent_task_weight_arg_binding_plan",
    ]
    if cuda_probe.get("status") == "pass":
        implemented_contracts.append(
            "cuda_full_weight_residency_probe"
            if cuda_probe.get("mode") == "full_residency"
            else "cuda_device_weight_copy_probe"
        )
    status = (
        "binding_plan_ready"
        if summary["binding_mismatch_count"] == 0
        and summary["metadata_status"] == "metadata_validated"
        else "binding_plan_incomplete"
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_weight_binding",
        "status": status,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_json": repo_relative(index_json),
        "weight_inventory_json": summary["inventory_source"],
        "metadata_json": summary["metadata_source"],
        "shard_dir": repo_relative(shard_dir),
        "metadata_status": summary["metadata_status"],
        "tensor_count": summary["index_tensor_count"],
        "planned_binding_count": summary["planned_binding_count"],
        "binding_mismatch_count": summary["binding_mismatch_count"],
        "binding_mismatches": summary["binding_mismatches"],
        "total_weight_bytes": summary["total_weight_bytes"],
        "cuda_probe": cuda_probe,
        "bindings": bindings,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": (
            [
                "persistent_task_weight_arg_runtime_binding",
                "qwen_kernel_weight_consumption",
            ]
            if cuda_probe.get("mode") == "full_residency"
            and cuda_probe.get("status") == "pass"
            else [
                "full_cuda_weight_residency",
                "persistent_task_weight_arg_runtime_binding",
                "qwen_kernel_weight_consumption",
            ]
        ),
    }

