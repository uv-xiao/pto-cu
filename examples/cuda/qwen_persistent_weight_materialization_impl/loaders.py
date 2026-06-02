"""Input artifact loading and pointer-table normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    DEFAULT_WEIGHT_ARGS,
    DEFAULT_WEIGHT_BINDING,
    ROOT,
    load_json,
    load_python_payload,
    repo_relative,
)


def load_or_build_weight_args(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        return load_json(path), repo_relative(path)
    if DEFAULT_WEIGHT_ARGS.is_file():
        payload = load_json(DEFAULT_WEIGHT_ARGS)
        if weight_args_shape_fields_ready(payload):
            return payload, repo_relative(DEFAULT_WEIGHT_ARGS)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_persistent_weight_args.py",
        "qwen_persistent_weight_args",
        "build_weight_arg_manifest",
    )
    return payload, "generated_from_examples/cuda/qwen_persistent_weight_args.py"


def weight_args_shape_fields_ready(payload: dict[str, Any]) -> bool:
    descriptors = payload.get("task_arg_descriptors", [])
    if not isinstance(descriptors, list):
        return False
    return any(logits_shape_fields_ready(item) for item in descriptors)


def logits_shape_fields_ready(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("callable") != "qwen_logits":
        return False
    fields = item.get("task_shape_fields")
    if not isinstance(fields, dict):
        return False
    return (
        int(fields.get("cols", 0)) > 0
        and int(fields.get("inner", 0)) > 0
        and float(fields.get("scalar1", 0.0)) > 0.0
    )


def load_or_build_weight_binding(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        return load_json(path), repo_relative(path)
    if DEFAULT_WEIGHT_BINDING.is_file():
        return load_json(DEFAULT_WEIGHT_BINDING), repo_relative(DEFAULT_WEIGHT_BINDING)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py",
        "qwen_cuda_weight_binding",
        "build_weight_binding",
        no_cuda_probe=True,
    )
    return payload, "generated_from_examples/cuda/qwen_cuda_weight_binding.py"


def parse_device_ptr(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        text = value.strip()
        if text:
            parsed = int(text, 0)
            if parsed > 0:
                return parsed
    return None


def pointer_map(pointer_table: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if pointer_table is None:
        return {}
    pointers = pointer_table.get("pointers", [])
    if not isinstance(pointers, list):
        raise ValueError("pointer table has no pointers list")
    mapped = {}
    for item in pointers:
        if not isinstance(item, dict) or not isinstance(item.get("slot_id"), int):
            raise ValueError("pointer table contains a malformed pointer record")
        dev_ptr = parse_device_ptr(item.get("device_ptr", item.get("device_ptr_hex")))
        if dev_ptr is None:
            raise ValueError(f"pointer slot {item['slot_id']} has no device pointer")
        mapped[item["slot_id"]] = {
            **item,
            "device_ptr": dev_ptr,
            "device_ptr_hex": f"0x{dev_ptr:x}",
        }
    return mapped


def binding_map(weight_binding: dict[str, Any]) -> dict[int, dict[str, Any]]:
    bindings = weight_binding.get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("weight binding artifact has no bindings list")
    mapped = {}
    for item in bindings:
        if not isinstance(item, dict) or not isinstance(item.get("slot_id"), int):
            raise ValueError("weight binding contains a malformed binding record")
        mapped[item["slot_id"]] = item
    return mapped
