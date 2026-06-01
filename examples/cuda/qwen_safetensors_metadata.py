#!/usr/bin/env python3
"""Probe Qwen safetensors shard headers against the expected weight contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX = (
    ROOT / "tmp" / "sources" / "qwen3-8b-model-safetensors-index-d117af2f.json"
)
DEFAULT_SHARD_DIR = ROOT / "tmp" / "sources" / "qwen3-8b-safetensors"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"

DTYPE_ALIASES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
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


def load_qwen_weight_inventory(index_json: Path) -> dict[str, Any]:
    module_path = SCRIPT_DIR / "qwen_weight_inventory.py"
    spec = importlib.util.spec_from_file_location("qwen_weight_inventory", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_weight_inventory(index_json)


def read_safetensors_header(path: Path) -> dict[str, Any]:
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
    return header


def expected_tensors(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = inventory.get("weight_shape_contract", {})
    tensors = contract.get("tensor_shapes", [])
    if not isinstance(tensors, list):
        raise ValueError("weight inventory has no tensor_shapes list")
    expected = {}
    for item in tensors:
        if not isinstance(item, dict):
            raise ValueError("weight inventory tensor_shapes contains a non-object")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("weight inventory tensor shape has no tensor name")
        expected[name] = item
    return expected


def normalize_dtype(dtype: Any) -> str:
    text = str(dtype)
    return DTYPE_ALIASES.get(text, text)


def compare_tensor(
    *,
    tensor_name: str,
    shard_name: str,
    header_entry: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> dict[str, Any]:
    if header_entry is None:
        return {
            "tensor": tensor_name,
            "shard": shard_name,
            "status": "missing_header_tensor",
        }
    if expected is None:
        return {
            "tensor": tensor_name,
            "shard": shard_name,
            "status": "missing_expected_contract",
            "actual_shape": header_entry.get("shape"),
            "actual_dtype": normalize_dtype(header_entry.get("dtype")),
        }
    actual_shape = header_entry.get("shape")
    actual_dtype = normalize_dtype(header_entry.get("dtype"))
    expected_shape = expected.get("shape")
    expected_dtype = expected.get("dtype")
    matches = actual_shape == expected_shape and actual_dtype == expected_dtype
    return {
        "tensor": tensor_name,
        "shard": shard_name,
        "status": "match" if matches else "mismatch",
        "actual_shape": actual_shape,
        "expected_shape": expected_shape,
        "actual_dtype": actual_dtype,
        "expected_dtype": expected_dtype,
    }


def build_metadata_probe(
    *,
    index_json: Path = DEFAULT_INDEX,
    weight_inventory_json: Path | None = None,
    shard_dir: Path = DEFAULT_SHARD_DIR,
) -> dict[str, Any]:
    index = load_json(index_json)
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError(f"weight_map is not an object in {index_json}")
    inventory = (
        load_json(weight_inventory_json)
        if weight_inventory_json is not None
        else load_qwen_weight_inventory(index_json)
    )
    expected = expected_tensors(inventory)
    shard_names = sorted({str(shard) for shard in weight_map.values()})
    missing_shards = [
        {
            "name": name,
            "path": repo_relative(shard_dir / name),
        }
        for name in shard_names
        if not (shard_dir / name).is_file()
    ]
    opened_headers: dict[str, dict[str, Any]] = {}
    shard_summaries = []
    for name in shard_names:
        path = shard_dir / name
        if not path.is_file():
            continue
        header = read_safetensors_header(path)
        tensor_entries = {
            key: value
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict)
        }
        opened_headers[name] = tensor_entries
        shard_summaries.append(
            {
                "name": name,
                "path": repo_relative(path),
                "header_tensor_count": len(tensor_entries),
            }
        )

    comparisons = []
    for tensor_name, shard_name in sorted(weight_map.items()):
        shard_tensors = opened_headers.get(str(shard_name))
        if shard_tensors is None:
            continue
        header_entry = shard_tensors.get(tensor_name)
        comparisons.append(
            compare_tensor(
                tensor_name=tensor_name,
                shard_name=str(shard_name),
                header_entry=header_entry,
                expected=expected.get(tensor_name),
            )
        )
    mismatches = [item for item in comparisons if item["status"] != "match"]
    opened_shard_count = len(opened_headers)
    validated_tensor_count = sum(1 for item in comparisons if item["status"] == "match")
    status = (
        "metadata_validated"
        if not missing_shards
        and not mismatches
        and validated_tensor_count == len(weight_map)
        else "shards_missing"
        if missing_shards
        else "metadata_mismatch"
    )
    implemented_contracts = []
    if opened_shard_count:
        implemented_contracts.append("safetensors_header_parse")
    if status == "metadata_validated":
        implemented_contracts.append("actual_safetensors_shape_dtype_validation")
    remaining_runtime_gaps = [
        "cuda_device_weight_binding",
        "persistent_task_weight_arg_binding",
    ]
    if status != "metadata_validated":
        remaining_runtime_gaps = [
            "safetensors_tensor_open",
            "actual_safetensors_shape_dtype_validation",
            *remaining_runtime_gaps,
        ]

    return {
        "schema_version": 1,
        "kind": "pto_qwen_safetensors_metadata_probe",
        "status": status,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "index_json": repo_relative(index_json),
        "weight_inventory_json": (
            repo_relative(weight_inventory_json)
            if weight_inventory_json is not None
            else "generated_from_examples/cuda/qwen_weight_inventory.py"
        ),
        "shard_dir": repo_relative(shard_dir),
        "expected_shard_count": len(shard_names),
        "opened_shard_count": opened_shard_count,
        "missing_shard_count": len(missing_shards),
        "missing_shards": missing_shards,
        "expected_tensor_count": len(weight_map),
        "validated_tensor_count": validated_tensor_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
        "shards": shard_summaries,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": remaining_runtime_gaps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-json", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--weight-inventory-json", type=Path)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_metadata_probe(
        index_json=args.index_json,
        weight_inventory_json=args.weight_inventory_json,
        shard_dir=args.shard_dir,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
