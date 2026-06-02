"""Top-level Qwen persistent weight-argument manifest builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    ABI_PATH,
    MODEL_ID,
    MODEL_REVISION,
    TENSOR_ARG_CAPACITY,
)
from .descriptors import build_task_descriptors
from .loaders import binding_map, load_or_build_weight_binding
from .shape_contract import QWEN3_8B_TASK_SHAPE, shape_contract_payload


def build_weight_arg_manifest(
    *,
    weight_binding_json: Path | None = None,
    num_hidden_layers: int = 36,
) -> dict[str, Any]:
    weight_binding, source = load_or_build_weight_binding(weight_binding_json)
    bindings = binding_map(weight_binding)
    descriptors = build_task_descriptors(
        bindings=bindings,
        num_hidden_layers=num_hidden_layers,
    )
    missing = sorted(
        {
            tensor
            for item in descriptors
            for tensor in item.get("missing_tensors", [])
        }
    )
    covered = sorted(
        {
            arg["tensor"]
            for item in descriptors
            for arg in item["tensor_args"]
            if "slot_id" in arg
        }
    )
    uncovered_bindings = sorted(set(bindings) - set(covered))
    max_tensor_args = max(
        (item["tensor_arg_count"] for item in descriptors),
        default=0,
    )
    capacity_ok = max_tensor_args <= TENSOR_ARG_CAPACITY
    complete = not missing and not uncovered_bindings and capacity_ok
    implemented_contracts = [
        "qwen_weight_task_decomposition",
        "qwen_task_shape_field_contract",
        "qwen_weight_tensor_metadata_contract",
        "persistent_dag_tensor_arg_capacity_check",
    ]
    if complete:
        implemented_contracts.append("persistent_task_weight_arg_binding_manifest")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_weight_args",
        "status": (
            "persistent_weight_args_ready"
            if complete
            else "persistent_weight_args_incomplete"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weight_binding_json": source,
        "weight_binding_status": weight_binding.get("status"),
        "weight_cuda_probe": weight_binding.get("cuda_probe", {}),
        "abi": {
            "task_struct": "PtoCudaPersistentDagTask",
            "source": ABI_PATH,
            "tensor_arg_field": "tensor_args",
            "tensor_arg_capacity": TENSOR_ARG_CAPACITY,
        },
        "num_hidden_layers": num_hidden_layers,
        "task_shape_contract": shape_contract_payload(QWEN3_8B_TASK_SHAPE),
        "task_arg_descriptor_count": len(descriptors),
        "task_arg_descriptors": descriptors,
        "covered_tensor_count": len(covered),
        "missing_tensor_count": len(missing),
        "missing_tensors": missing,
        "uncovered_binding_count": len(uncovered_bindings),
        "uncovered_bindings": uncovered_bindings[:32],
        "max_tensor_args_per_task": max_tensor_args,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "persistent_task_weight_arg_runtime_binding",
            "qwen_kernel_weight_consumption",
        ],
    }
