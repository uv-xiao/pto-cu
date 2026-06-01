"""Safetensors metadata and binding-slot planning."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from .common import ROOT, load_json, load_python_payload, repo_relative


DTYPE_ALIASES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"{path} is too short for a safetensors header")
        header_size = struct.unpack("<Q", size_bytes)[0]
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{path} ended before safetensors header finished")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header is not a JSON object")
    return header_size, header


def normalize_dtype(dtype: Any) -> str:
    return DTYPE_ALIASES.get(str(dtype), str(dtype))

def load_or_build_inventory(
    *,
    index_json: Path,
    weight_inventory_json: Path | None,
) -> tuple[dict[str, Any], str]:
    if weight_inventory_json is not None:
        return load_json(weight_inventory_json), repo_relative(weight_inventory_json)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_weight_inventory.py",
        "qwen_weight_inventory",
        "build_weight_inventory",
        index_json,
    )
    return payload, "generated_from_examples/cuda/qwen_weight_inventory.py"


def load_or_build_metadata(
    *,
    index_json: Path,
    weight_inventory_json: Path | None,
    metadata_json: Path | None,
    shard_dir: Path,
) -> tuple[dict[str, Any], str]:
    if metadata_json is not None:
        return load_json(metadata_json), repo_relative(metadata_json)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_safetensors_metadata.py",
        "qwen_safetensors_metadata",
        "build_metadata_probe",
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
        shard_dir=shard_dir,
    )
    return payload, "generated_from_examples/cuda/qwen_safetensors_metadata.py"


def inventory_tensor_contracts(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = inventory.get("weight_shape_contract", {})
    tensors = contract.get("tensor_shapes", [])
    if not isinstance(tensors, list):
        raise ValueError("weight inventory has no tensor_shapes list")
    return {
        item["name"]: item
        for item in tensors
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def inventory_group_map(inventory: dict[str, Any]) -> dict[str, str]:
    groups = {}
    for group in inventory.get("binding_groups", []):
        if not isinstance(group, dict):
            continue
        group_id = group.get("id")
        if not isinstance(group_id, str):
            continue
        for tensor in group.get("sample_tensors", []):
            if isinstance(tensor, str):
                groups[tensor] = group_id
    return groups


def binding_group_for_tensor(
    tensor_name: str,
    known_groups: dict[str, str],
) -> str:
    if tensor_name in known_groups:
        return known_groups[tensor_name]
    if tensor_name.startswith("model.embed_tokens."):
        return "embedding"
    if any(
        marker in tensor_name
        for marker in (
            ".self_attn.q_proj.",
            ".self_attn.k_proj.",
            ".self_attn.v_proj.",
            ".self_attn.o_proj.",
        )
    ):
        return "attention_qkv_o"
    if any(
        marker in tensor_name
        for marker in (
            ".input_layernorm.",
            ".post_attention_layernorm.",
            ".self_attn.q_norm.",
            ".self_attn.k_norm.",
        )
    ):
        return "attention_norms"
    if any(
        marker in tensor_name
        for marker in (
            ".mlp.gate_proj.",
            ".mlp.up_proj.",
            ".mlp.down_proj.",
        )
    ):
        return "mlp_gate_up_down"
    if tensor_name.startswith("model.norm.") or tensor_name.startswith("lm_head."):
        return "norm_and_logits"
    return "unclassified"


def build_bindings(
    *,
    index_json: Path,
    weight_inventory_json: Path | None,
    metadata_json: Path | None,
    shard_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = load_json(index_json)
    inventory, inventory_source = load_or_build_inventory(
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
    )
    metadata, metadata_source = load_or_build_metadata(
        index_json=index_json,
        weight_inventory_json=weight_inventory_json,
        metadata_json=metadata_json,
        shard_dir=shard_dir,
    )
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError(f"weight_map is not an object in {index_json}")

    expected = inventory_tensor_contracts(inventory)
    known_groups = inventory_group_map(inventory)
    opened_headers = {}
    for shard_name in sorted({str(shard) for shard in weight_map.values()}):
        path = shard_dir / shard_name
        if not path.is_file():
            continue
        header_size, header = read_safetensors_header(path)
        opened_headers[shard_name] = {
            "path": path,
            "header_size": header_size,
            "data_base_offset": 8 + header_size,
            "tensors": {
                key: value
                for key, value in header.items()
                if key != "__metadata__" and isinstance(value, dict)
            },
        }

    bindings = []
    mismatches = []
    for slot_id, (tensor_name, shard_name_any) in enumerate(sorted(weight_map.items())):
        shard_name = str(shard_name_any)
        shard = opened_headers.get(shard_name)
        header_entry = shard["tensors"].get(tensor_name) if shard else None
        expected_entry = expected.get(tensor_name)
        if shard is None or header_entry is None or expected_entry is None:
            mismatches.append(tensor_name)
            continue
        offsets = header_entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(item, int) for item in offsets)
            or offsets[1] < offsets[0]
        ):
            mismatches.append(tensor_name)
            continue
        dtype = normalize_dtype(header_entry.get("dtype"))
        shape = header_entry.get("shape")
        if shape != expected_entry.get("shape") or dtype != expected_entry.get("dtype"):
            mismatches.append(tensor_name)
            continue
        absolute_offsets = [
            shard["data_base_offset"] + offsets[0],
            shard["data_base_offset"] + offsets[1],
        ]
        bindings.append(
            {
                "slot_id": slot_id,
                "tensor": tensor_name,
                "shard": shard_name,
                "shard_path": repo_relative(shard["path"]),
                "dtype": dtype,
                "shape": shape,
                "size_bytes": offsets[1] - offsets[0],
                "file_data_offsets": offsets,
                "file_absolute_offsets": absolute_offsets,
                "binding_group": binding_group_for_tensor(tensor_name, known_groups),
                "persistent_arg_role": "readonly_weight_tensor",
                "cuda_binding_state": "planned_not_resident",
            }
        )

    summary = {
        "metadata_status": metadata.get("status"),
        "index_tensor_count": len(weight_map),
        "planned_binding_count": len(bindings),
        "binding_mismatch_count": len(mismatches),
        "binding_mismatches": mismatches[:32],
        "total_weight_bytes": sum(item["size_bytes"] for item in bindings),
        "inventory_source": inventory_source,
        "metadata_source": metadata_source,
    }
    return summary, bindings

