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


def build_weight_inventory(index_json: Path = DEFAULT_INDEX) -> dict[str, Any]:
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
    return {
        "schema_version": 1,
        "kind": "pto_qwen_weight_inventory",
        "status": "partial_inventory",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_json": repo_relative(index_json),
        "total_size_bytes": payload.get("metadata", {}).get("total_size"),
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
        "implemented_contracts": [
            "safetensors_index_parse",
            "weight_tensor_grouping",
            "weight_shard_inventory",
        ],
        "remaining_runtime_gaps": [
            "safetensors_tensor_open",
            "tensor_shape_dtype_validation",
            "cuda_device_weight_binding",
            "persistent_task_weight_arg_binding",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_weight_inventory(args.index_json)
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
