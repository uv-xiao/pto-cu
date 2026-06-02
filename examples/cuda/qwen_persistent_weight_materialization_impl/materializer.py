"""Descriptor-level materialization of persistent Qwen weight arguments."""

from __future__ import annotations

from typing import Any


def materialized_tensor_arg(
    *,
    arg: dict[str, Any],
    bindings: dict[int, dict[str, Any]],
    pointers: dict[int, dict[str, Any]],
    pointer_table_ready: bool,
) -> dict[str, Any]:
    slot_id = arg["slot_id"]
    binding = bindings.get(slot_id, {})
    record = {
        "arg": arg["arg"],
        "slot_id": slot_id,
        "tensor": arg["tensor"],
    }
    pointer = pointers.get(slot_id)
    if pointer_table_ready and pointer is None:
        return {**record, "status": "missing_resident_pointer"}
    if pointer is None:
        return {
            **record,
            "device_ptr_source": f"resident_weight_ptrs[{slot_id}]",
            "size_bytes": binding.get("size_bytes"),
            "status": "requires_live_pointer",
        }
    if pointer.get("tensor") not in {None, arg["tensor"]}:
        return {**record, "status": "pointer_tensor_mismatch"}
    return {
        **record,
        "device_ptr": pointer["device_ptr"],
        "device_ptr_hex": pointer["device_ptr_hex"],
        "size_bytes": int(pointer.get("size_bytes", binding.get("size_bytes", 0))),
    }


def materialized_descriptor(
    *,
    descriptor: dict[str, Any],
    bindings: dict[int, dict[str, Any]],
    pointers: dict[int, dict[str, Any]],
    pointer_table_ready: bool,
) -> dict[str, Any]:
    tensor_args = [
        materialized_tensor_arg(
            arg=arg,
            bindings=bindings,
            pointers=pointers,
            pointer_table_ready=pointer_table_ready,
        )
        for arg in descriptor.get("tensor_args", [])
    ]
    missing = [
        item["tensor"]
        for item in tensor_args
        if item.get("status") in {
            "missing_resident_pointer",
            "pointer_tensor_mismatch",
        }
    ]
    record = {
        "id": descriptor["id"],
        "callable": descriptor["callable"],
        "phase": descriptor["phase"],
        "tensor_arg_count": len(tensor_args),
        "tensor_args": tensor_args,
        "status": "ready" if not missing else "missing_resident_pointer",
        "missing_tensors": missing,
    }
    for key in ("scalar_fields", "task_shape_fields"):
        if isinstance(descriptor.get(key), dict):
            record[key] = dict(descriptor[key])
    return record
