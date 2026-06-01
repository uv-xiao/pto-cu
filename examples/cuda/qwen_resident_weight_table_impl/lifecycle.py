"""Build Qwen resident weight table lifecycle artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_resident_weight_table_impl.common import (
    DEFAULT_COPY_CHUNK_BYTES,
    DEFAULT_HOST_RUNTIME,
    MODEL_ID,
    MODEL_REVISION,
    load_or_build_weight_binding,
    load_weight_args_path,
    repo_relative,
    write_json,
)
from qwen_resident_weight_table_impl.materialization import (
    materialize_with_pointer_table,
)
from qwen_resident_weight_table_impl.resident import (
    ResidentWeightTableOwner,
    dry_run_owner,
)


def build_owner(
    *,
    bindings: list[dict[str, Any]],
    dry_run: bool,
    device: int,
    host_runtime: Path,
    pointer_base: int,
    pointer_stride: int,
    copy_chunk_bytes: int,
) -> ResidentWeightTableOwner:
    if dry_run:
        return dry_run_owner(
            bindings=bindings,
            pointer_base=pointer_base,
            pointer_stride=pointer_stride,
        )
    from qwen_resident_weight_table_impl.cuda_live import (
        CudaResidentWeightTableOwner,
    )

    return CudaResidentWeightTableOwner(
        bindings=bindings,
        device=device,
        host_runtime=host_runtime,
        copy_chunk_bytes=copy_chunk_bytes,
    )


def build_resident_table_lifecycle(
    *,
    weight_binding_json: Path | None = None,
    weight_args_json: Path | None = None,
    dry_run: bool = True,
    device: int = 0,
    host_runtime: Path = DEFAULT_HOST_RUNTIME,
    pointer_base: int = 0x50000000,
    pointer_stride: int = 0x100000,
    copy_chunk_bytes: int = DEFAULT_COPY_CHUNK_BYTES,
) -> dict[str, Any]:
    weight_binding, binding_source = load_or_build_weight_binding(weight_binding_json)
    bindings = weight_binding.get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("weight binding artifact has no bindings list")
    resolved_weight_args = load_weight_args_path(weight_args_json)
    owner = build_owner(
        bindings=bindings,
        dry_run=dry_run,
        device=device,
        host_runtime=host_runtime,
        pointer_base=pointer_base,
        pointer_stride=pointer_stride,
        copy_chunk_bytes=copy_chunk_bytes,
    )
    with owner:
        pointer_table = owner.pointer_table()
        materialization = (
            materialize_with_pointer_table(
                weight_args_json=resolved_weight_args,
                weight_binding_json=weight_binding_json,
                pointer_table=pointer_table,
            )
            if resolved_weight_args is not None
            else {}
        )
    closed = owner.pointer_table()
    return lifecycle_payload(
        dry_run=dry_run,
        binding_source=binding_source,
        weight_args_path=resolved_weight_args,
        pointer_table=pointer_table,
        closed_pointer_table=closed,
        materialization=materialization,
    )


def lifecycle_payload(
    *,
    dry_run: bool,
    binding_source: str,
    weight_args_path: Path | None,
    pointer_table: dict[str, Any],
    closed_pointer_table: dict[str, Any],
    materialization: dict[str, Any],
) -> dict[str, Any]:
    status_ready = (
        pointer_table.get("status") == "resident_weight_pointer_table_ready"
        and closed_pointer_table.get("status")
        == "resident_weight_pointer_table_closed"
        and closed_pointer_table.get("freed_pointer_count")
        == pointer_table.get("pointer_count")
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_resident_weight_table_lifecycle",
        "status": (
            "resident_weight_table_lifecycle_ready"
            if status_ready
            else "resident_weight_table_lifecycle_incomplete"
        ),
        "mode": "dry_run_pointer_lifecycle" if dry_run else "cuda_live",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_binding_json": binding_source,
        "weight_args_json": repo_relative(weight_args_path) if weight_args_path else None,
        "pointer_table": pointer_table,
        "closed_pointer_table": closed_pointer_table,
        "materialization_status": materialization.get("status", "not_run"),
        "materialized_task_count": materialization.get("materialized_task_count", 0),
        "bound_tensor_pointer_count": materialization.get(
            "bound_tensor_pointer_count",
            0,
        ),
        "implemented_contracts": [
            "live_resident_weight_table_owner",
            "resident_pointer_table_materialization_bridge",
            "dry_run_pointer_lifecycle" if dry_run else "cuda_live_weight_table",
        ],
        "remaining_runtime_gaps": [
            "decode_loop_table_integration",
            "qwen_kernel_weight_consumption",
        ],
    }
