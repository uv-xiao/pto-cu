#!/usr/bin/env python3
"""Emit Qwen safetensors weight-inventory evidence for PTO CUDA serving."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = (
    ROOT / "tmp" / "sources" / "qwen3-8b-model-safetensors-index-d117af2f.json"
)
DEFAULT_CONFIG = ROOT / "tmp" / "sources" / "qwen3-8b-config-d117af2f.json"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"


BINDING_GROUP_PREFIXES = {
    "embedding": ("model.embed_tokens.",),
    "attention_qkv_o": (
        ".self_attn.q_proj.",
        ".self_attn.k_proj.",
        ".self_attn.v_proj.",
        ".self_attn.o_proj.",
    ),
    "attention_norms": (
        ".input_layernorm.",
        ".post_attention_layernorm.",
        ".self_attn.q_norm.",
        ".self_attn.k_norm.",
    ),
    "mlp_gate_up_down": (
        ".mlp.gate_proj.",
        ".mlp.up_proj.",
        ".mlp.down_proj.",
    ),
    "norm_and_logits": ("model.norm.", "lm_head."),
}
D_TYPE_SIZES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


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


def group_for_tensor(name: str) -> str:
    for group, markers in BINDING_GROUP_PREFIXES.items():
        if any(marker in name or name.startswith(marker) for marker in markers):
            return group
    return "unclassified"


def binding_groups(weight_map: dict[str, str]) -> list[dict[str, Any]]:
    by_group: dict[str, list[str]] = {
        group: [] for group in [*BINDING_GROUP_PREFIXES, "unclassified"]
    }
    for tensor_name in sorted(weight_map):
        by_group[group_for_tensor(tensor_name)].append(tensor_name)
    groups = []
    for group, tensors in by_group.items():
        groups.append(
            {
                "id": group,
                "tensor_count": len(tensors),
                "sample_tensors": tensors[:8],
                "status": (
                    "planned_loader_binding" if tensors else "not_present"
                ),
            }
        )
    return groups


def int_config(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Qwen config has non-integer {key}: {value!r}")
    return value


def expected_tensor_shape(name: str, config: dict[str, Any]) -> list[int] | None:
    hidden_size = int_config(config, "hidden_size")
    intermediate_size = int_config(config, "intermediate_size")
    vocab_size = int_config(config, "vocab_size")
    attention_heads = int_config(config, "num_attention_heads")
    kv_heads = int_config(config, "num_key_value_heads")
    head_dim = int_config(config, "head_dim")

    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return [vocab_size, hidden_size]
    if name == "model.norm.weight":
        return [hidden_size]
    if name.endswith(".input_layernorm.weight"):
        return [hidden_size]
    if name.endswith(".post_attention_layernorm.weight"):
        return [hidden_size]
    if name.endswith(".self_attn.q_norm.weight"):
        return [head_dim]
    if name.endswith(".self_attn.k_norm.weight"):
        return [head_dim]
    if name.endswith(".self_attn.q_proj.weight"):
        return [attention_heads * head_dim, hidden_size]
    if name.endswith(".self_attn.k_proj.weight"):
        return [kv_heads * head_dim, hidden_size]
    if name.endswith(".self_attn.v_proj.weight"):
        return [kv_heads * head_dim, hidden_size]
    if name.endswith(".self_attn.o_proj.weight"):
        return [hidden_size, attention_heads * head_dim]
    if name.endswith(".mlp.gate_proj.weight"):
        return [intermediate_size, hidden_size]
    if name.endswith(".mlp.up_proj.weight"):
        return [intermediate_size, hidden_size]
    if name.endswith(".mlp.down_proj.weight"):
        return [hidden_size, intermediate_size]
    return None


def element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def build_shape_contract(
    *,
    config_json: Path,
    weight_map: dict[str, str],
    index_total_size_bytes: int | None,
) -> dict[str, Any]:
    if not config_json.is_file():
        return {
            "status": "config_missing",
            "config_json": repo_relative(config_json),
            "remaining_gap": "expected_shape_dtype_contract",
        }

    config = load_json(config_json)
    dtype = str(config.get("torch_dtype", ""))
    bytes_per_element = D_TYPE_SIZES.get(dtype)
    if bytes_per_element is None:
        raise ValueError(f"Unsupported Qwen torch_dtype in {config_json}: {dtype}")

    tensor_shapes = []
    unknown_tensors = []
    total_bytes = 0
    for name in sorted(weight_map):
        shape = expected_tensor_shape(name, config)
        if shape is None:
            unknown_tensors.append(name)
            continue
        bytes_for_tensor = element_count(shape) * bytes_per_element
        total_bytes += bytes_for_tensor
        tensor_shapes.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "expected_size_bytes": bytes_for_tensor,
            }
        )

    complete = not unknown_tensors and len(tensor_shapes) == len(weight_map)
    size_matches = (
        index_total_size_bytes == total_bytes
        if index_total_size_bytes is not None
        else False
    )
    return {
        "status": (
            "complete_for_index"
            if complete and size_matches
            else "incomplete_or_size_mismatch"
        ),
        "config_json": repo_relative(config_json),
        "dtype": dtype,
        "bytes_per_element": bytes_per_element,
        "tensor_count": len(tensor_shapes),
        "unknown_tensor_count": len(unknown_tensors),
        "unknown_tensors": unknown_tensors,
        "expected_total_size_bytes": total_bytes,
        "index_total_size_bytes": index_total_size_bytes,
        "size_matches_index": size_matches,
        "tensor_shapes": tensor_shapes,
    }


def build_weight_inventory(
    index_json: Path = DEFAULT_INDEX,
    config_json: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if not index_json.is_file():
        return {
            "schema_version": 1,
            "kind": "pto_qwen_weight_inventory",
            "status": "index_missing",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "index_json": repo_relative(index_json),
            "remaining_runtime_gaps": [
                "safetensors_index_capture",
                "safetensors_tensor_open",
                "cuda_device_weight_binding",
            ],
        }

    payload = load_json(index_json)
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError(f"weight_map is not an object in {index_json}")
    shard_counter = Counter(str(shard) for shard in weight_map.values())
    index_total_size_bytes = payload.get("metadata", {}).get("total_size")
    if not isinstance(index_total_size_bytes, int):
        index_total_size_bytes = None
    shape_contract = build_shape_contract(
        config_json=config_json,
        weight_map=weight_map,
        index_total_size_bytes=index_total_size_bytes,
    )
    implemented_contracts = [
        "safetensors_index_parse",
        "weight_tensor_grouping",
        "weight_shard_inventory",
    ]
    if shape_contract.get("status") == "complete_for_index":
        implemented_contracts.append("expected_shape_dtype_contract")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_weight_inventory",
        "status": "partial_inventory",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_json": repo_relative(index_json),
        "config_json": repo_relative(config_json),
        "total_size_bytes": index_total_size_bytes,
        "tensor_count": len(weight_map),
        "shard_count": len(shard_counter),
        "shards": [
            {
                "name": name,
                "tensor_count": count,
            }
            for name, count in sorted(shard_counter.items())
        ],
        "binding_groups": binding_groups(weight_map),
        "weight_shape_contract": shape_contract,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "safetensors_tensor_open",
            "actual_safetensors_shape_dtype_validation",
            "cuda_device_weight_binding",
            "persistent_task_weight_arg_binding",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_weight_inventory(args.index_json, args.config_json)
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
