"""Top-level Qwen persistent weight materialization builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .abi import dag_task_abi
from .common import MODEL_ID, MODEL_REVISION, load_json, repo_relative
from .loaders import (
    binding_map,
    load_or_build_weight_args,
    load_or_build_weight_binding,
    pointer_map,
)
from .materializer import materialized_descriptor


def build_materialization_manifest(
    *,
    weight_args_json: Path | None = None,
    weight_binding_json: Path | None = None,
    pointer_table_json: Path | None = None,
    pointer_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    weight_args, weight_args_source = load_or_build_weight_args(weight_args_json)
    weight_binding, weight_binding_source = load_or_build_weight_binding(
        weight_binding_json
    )
    if pointer_table_json is not None and pointer_table is not None:
        raise ValueError("pass pointer_table_json or pointer_table, not both")
    loaded_pointer_table = (
        load_json(pointer_table_json) if pointer_table_json else pointer_table
    )
    pointer_source = (
        repo_relative(pointer_table_json)
        if pointer_table_json
        else "in_memory_pointer_table"
        if pointer_table is not None
        else None
    )
    pointers = pointer_map(loaded_pointer_table)
    bindings = binding_map(weight_binding)
    pointer_table_ready = (
        loaded_pointer_table is not None
        and loaded_pointer_table.get("status")
        == "resident_weight_pointer_table_ready"
    )
    descriptors = [
        materialized_descriptor(
            descriptor=descriptor,
            bindings=bindings,
            pointers=pointers,
            pointer_table_ready=pointer_table_ready,
        )
        for descriptor in weight_args.get("task_arg_descriptors", [])
    ]
    bound = [
        arg
        for descriptor in descriptors
        for arg in descriptor["tensor_args"]
        if "device_ptr" in arg
    ]
    missing = sorted(
        {
            tensor
            for descriptor in descriptors
            for tensor in descriptor["missing_tensors"]
        }
    )
    symbolic = [
        arg
        for descriptor in descriptors
        for arg in descriptor["tensor_args"]
        if arg.get("status") == "requires_live_pointer"
    ]
    complete = (
        weight_args.get("status") == "persistent_weight_args_ready"
        and pointer_table_ready
        and not missing
    )
    implemented_contracts = [
        "persistent_task_weight_arg_runtime_materializer",
        "ctypes_persistent_dag_task_layout",
        "runtime_generated_tensor_pointer_requirements",
    ]
    if complete:
        implemented_contracts.append("resident_weight_pointer_table_validation")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_weight_materialization",
        "status": (
            "persistent_weight_materialization_ready"
            if complete
            else "persistent_weight_materialization_plan_ready"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_args_json": weight_args_source,
        "weight_binding_json": weight_binding_source,
        "pointer_table_json": pointer_source,
        "weight_args_status": weight_args.get("status"),
        "weight_binding_status": weight_binding.get("status"),
        "pointer_table_status": (
            loaded_pointer_table.get("status")
            if loaded_pointer_table is not None
            else "not_supplied"
        ),
        "abi": dag_task_abi(),
        "materialized_task_count": len(descriptors),
        "materialized_task_descriptors": descriptors,
        "bound_tensor_pointer_count": len(bound),
        "symbolic_tensor_pointer_count": len(symbolic),
        "symbolic_tensor_pointer_requirements": symbolic[:16],
        "missing_pointer_count": len(missing),
        "missing_pointers": missing,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "live_decode_loop_pointer_table",
            "rope_table_live_pointer_binding",
            "qwen_kernel_weight_consumption",
        ],
    }
