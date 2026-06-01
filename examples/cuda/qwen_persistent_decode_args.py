#!/usr/bin/env python3
"""Bind Qwen CUDA token buffers into persistent decode task arguments."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUDA_TOKEN_BUFFER = (
    ROOT
    / "tmp"
    / "cuda-backend"
    / "pto-serving-token-buffer-2026-06-01"
    / "qwen-cuda-token-buffer-binding.json"
)
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "d117af2f304f02a8647f88fe05b61cfb405a1d9e"
TOKEN_FIELDS = {
    "a": "input_ids",
    "b": "attention_mask",
    "out": "output_ids",
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

def load_python_payload(
    path: Path,
    module_name: str,
    build_name: str,
) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, build_name)(no_cuda_probe=True)


def load_or_build_token_binding(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is not None:
        return load_json(path), repo_relative(path)
    if DEFAULT_CUDA_TOKEN_BUFFER.is_file():
        return (
            load_json(DEFAULT_CUDA_TOKEN_BUFFER),
            repo_relative(DEFAULT_CUDA_TOKEN_BUFFER),
        )
    payload = load_python_payload(
        ROOT / "examples" / "cuda" / "qwen_cuda_token_buffer_binding.py",
        "qwen_cuda_token_buffer_binding",
        "build_cuda_token_buffer_binding",
    )
    return payload, "generated_from_examples/cuda/qwen_cuda_token_buffer_binding.py"


def parse_device_ptr(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip():
        parsed = int(value, 0)
        return parsed if parsed > 0 else None
    return None


def token_pointer_map(
    pointer_table: dict[str, Any] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    if pointer_table is None:
        return {}
    mapped = {}
    for item in pointer_table.get("pointers", []):
        if not isinstance(item, dict):
            raise ValueError("token pointer table contains a malformed record")
        workload_id = item.get("workload_id")
        buffer = item.get("buffer")
        if not isinstance(workload_id, str) or not isinstance(buffer, str):
            raise ValueError("token pointer record has no workload_id or buffer")
        dev_ptr = parse_device_ptr(item.get("device_ptr", item.get("device_ptr_hex")))
        if dev_ptr is None:
            raise ValueError(f"token pointer {workload_id}:{buffer} has no pointer")
        mapped[(workload_id, buffer)] = {
            **item,
            "device_ptr": dev_ptr,
            "device_ptr_hex": f"0x{dev_ptr:x}",
        }
    return mapped


def dag_task_abi() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from simpler_setup.cuda_callable_compiler import CudaPersistentDagTask

    fields = {
        name: getattr(CudaPersistentDagTask, name).offset
        for name in [
            "a",
            "b",
            "out",
            "n",
            "rows",
            "cols",
            "inner",
            "tensor_args",
        ]
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
        "token_pointer_fields": TOKEN_FIELDS,
        "weight_pointer_field": "tensor_args",
    }


def buffer_descriptor(record: dict[str, Any], buffer: str) -> dict[str, Any]:
    return record[f"{buffer}_device_buffer"]


def pointer_binding(
    *,
    workload_id: str,
    field: str,
    buffer: str,
    descriptor: dict[str, Any],
    pointers: dict[tuple[str, str], dict[str, Any]],
    pointer_table_ready: bool,
) -> dict[str, Any]:
    pointer = pointers.get((workload_id, buffer))
    base = {
        "field": field,
        "buffer": buffer,
        "byte_count": descriptor["byte_count"],
    }
    if pointer_table_ready and pointer is None:
        return {**base, "status": "missing_token_pointer"}
    if pointer is None:
        return {
            **base,
            "device_ptr_source": f"cuda_token_buffers[{workload_id}].{buffer}",
            "status": "requires_live_pointer",
        }
    return {
        **base,
        "device_ptr": pointer["device_ptr"],
        "device_ptr_hex": pointer["device_ptr_hex"],
    }


def workload_decode_args(
    *,
    record: dict[str, Any],
    pointers: dict[tuple[str, str], dict[str, Any]],
    pointer_table_ready: bool,
) -> dict[str, Any]:
    workload_id = record["workload_id"]
    bindings = [
        pointer_binding(
            workload_id=workload_id,
            field=field,
            buffer=buffer,
            descriptor=buffer_descriptor(record, buffer),
            pointers=pointers,
            pointer_table_ready=pointer_table_ready,
        )
        for field, buffer in TOKEN_FIELDS.items()
    ]
    missing = [
        item["buffer"]
        for item in bindings
        if item.get("status") == "missing_token_pointer"
    ]
    symbolic = [
        item
        for item in bindings
        if item.get("status") == "requires_live_pointer"
    ]
    scalars = record["scalar_bindings"]
    return {
        "workload_id": workload_id,
        "status": "ready" if not missing and not symbolic else "plan_ready",
        "pointer_bindings": bindings,
        "missing_token_buffers": missing,
        "symbolic_token_buffer_count": len(symbolic),
        "scalar_fields": {
            "n": int(scalars["prompt_token_count"]),
            "rows": int(scalars["max_batch_size"]),
            "cols": int(scalars["decode_tokens"]),
            "inner": int(scalars["first_decode_position"]),
        },
    }


def build_decode_arg_manifest(
    *,
    cuda_token_buffer_json: Path | None = None,
    token_pointer_table_json: Path | None = None,
) -> dict[str, Any]:
    token_binding, token_binding_source = load_or_build_token_binding(
        cuda_token_buffer_json
    )
    pointer_table = (
        load_json(token_pointer_table_json) if token_pointer_table_json else None
    )
    pointer_source = (
        repo_relative(token_pointer_table_json)
        if token_pointer_table_json
        else None
    )
    pointer_table_ready = (
        pointer_table is not None
        and pointer_table.get("status") == "cuda_token_pointer_table_ready"
    )
    pointers = token_pointer_map(pointer_table)
    records = [
        workload_decode_args(
            record=record,
            pointers=pointers,
            pointer_table_ready=pointer_table_ready,
        )
        for record in token_binding.get("workload_records", [])
    ]
    ready = all(record["status"] == "ready" for record in records)
    implemented = [
        "persistent_decode_token_arg_binding",
        "base_pointer_fields_for_tokens",
        "weight_tensor_args_preserved",
    ]
    return {
        "schema_version": 1,
        "kind": "pto_qwen_persistent_decode_args",
        "status": (
            "persistent_decode_args_ready"
            if ready
            else "persistent_decode_args_plan_ready"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "cuda_token_buffer_json": token_binding_source,
        "token_pointer_table_json": pointer_source,
        "cuda_token_buffer_status": token_binding.get("status"),
        "token_pointer_table_status": (
            pointer_table.get("status") if pointer_table is not None else "not_supplied"
        ),
        "abi": dag_task_abi(),
        "workload_decode_args": records,
        "implemented_contracts": implemented,
        "remaining_runtime_gaps": [
            "numerically_correct_qwen_token_consumption",
            "decode_loop_execution",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-token-buffer-json", type=Path)
    parser.add_argument("--token-pointer-table-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_decode_arg_manifest(
        cuda_token_buffer_json=args.cuda_token_buffer_json,
        token_pointer_table_json=args.token_pointer_table_json,
    )
    if args.output_json:
        write_json(args.output_json, payload)
        print(repo_relative(args.output_json))
    else:
        print(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
