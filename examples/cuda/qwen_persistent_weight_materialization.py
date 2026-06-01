#!/usr/bin/env python3
"""Materialize Qwen persistent DAG weight descriptors with resident pointers."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHT_ARGS = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-weight-args-21589e81"
    / "qwen-persistent-weight-args.json"
)
DEFAULT_WEIGHT_BINDING = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-weight-residency-1ae913c9"
    / "qwen-cuda-weight-residency.json"
)
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"


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


def load_python_payload(
    path: Path,
    module_name: str,
    build_name: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, build_name)(*args, **kwargs)


def load_or_build_weight_args(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        return load_json(path), repo_relative(path)
    if DEFAULT_WEIGHT_ARGS.is_file():
        return load_json(DEFAULT_WEIGHT_ARGS), repo_relative(DEFAULT_WEIGHT_ARGS)
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_persistent_weight_args.py",
        "qwen_persistent_weight_args",
        "build_weight_arg_manifest",
    )
    return payload, "generated_from_examples/cuda/qwen_persistent_weight_args.py"


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


def dag_task_abi() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from simpler_setup.cuda_callable_compiler import CudaPersistentDagTask

    fields = {
        "func_id": CudaPersistentDagTask.func_id.offset,
        "tensor_args": CudaPersistentDagTask.tensor_args.offset,
        "scalar_args": CudaPersistentDagTask.scalar_args.offset,
        "tensor_arg_count": CudaPersistentDagTask.tensor_arg_count.offset,
        "scalar_arg_count": CudaPersistentDagTask.scalar_arg_count.offset,
    }
    return {
        "task_struct": "CudaPersistentDagTask",
        "c_header_struct": "PtoCudaPersistentDagTask",
        "python_source": "simpler_setup/cuda_callable_compiler.py",
        "c_header_source": (
            "src/cuda/platform/include/host/"
            "pto_cuda_persistent_device_abi.h"
        ),
        "sizeof_bytes": ctypes.sizeof(CudaPersistentDagTask),
        "field_offsets": fields,
        "tensor_arg_capacity": 4,
    }


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
    return {
        "id": descriptor["id"],
        "callable": descriptor["callable"],
        "phase": descriptor["phase"],
        "tensor_arg_count": len(tensor_args),
        "tensor_args": tensor_args,
        "status": "ready" if not missing else "missing_resident_pointer",
        "missing_tensors": missing,
    }


def build_materialization_manifest(
    *,
    weight_args_json: Path | None = None,
    weight_binding_json: Path | None = None,
    pointer_table_json: Path | None = None,
) -> dict[str, Any]:
    weight_args, weight_args_source = load_or_build_weight_args(weight_args_json)
    weight_binding, weight_binding_source = load_or_build_weight_binding(
        weight_binding_json
    )
    pointer_table = load_json(pointer_table_json) if pointer_table_json else None
    pointer_source = repo_relative(pointer_table_json) if pointer_table_json else None
    pointers = pointer_map(pointer_table)
    bindings = binding_map(weight_binding)
    pointer_table_ready = (
        pointer_table is not None
        and pointer_table.get("status") == "resident_weight_pointer_table_ready"
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
            pointer_table.get("status") if pointer_table is not None else "not_supplied"
        ),
        "abi": dag_task_abi(),
        "materialized_task_count": len(descriptors),
        "materialized_task_descriptors": descriptors,
        "bound_tensor_pointer_count": len(bound),
        "symbolic_tensor_pointer_count": len(symbolic),
        "missing_pointer_count": len(missing),
        "missing_pointers": missing,
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "live_decode_loop_pointer_table",
            "qwen_kernel_weight_consumption",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-args-json", type=Path)
    parser.add_argument("--weight-binding-json", type=Path)
    parser.add_argument("--pointer-table-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_materialization_manifest(
        weight_args_json=args.weight_args_json,
        weight_binding_json=args.weight_binding_json,
        pointer_table_json=args.pointer_table_json,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
