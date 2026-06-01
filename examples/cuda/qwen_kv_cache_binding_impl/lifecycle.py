"""Build Qwen KV-cache pointer lifecycle artifacts."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
LIFECYCLE_PLAN = ROOT / "examples" / "cuda" / "qwen_serving_lifecycle_plan.py"
KV_FIELDS = {"c": "key_cache", "d": "value_cache"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_lifecycle_plan() -> dict[str, Any]:
    module = load_module(LIFECYCLE_PLAN, "qwen_lifecycle_for_kv_cache")
    return module.build_lifecycle_plan()


def dag_task_abi() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from simpler_setup.cuda_callable_compiler import CudaPersistentDagTask

    return {
        "task_struct": "CudaPersistentDagTask",
        "c_header_struct": "PtoCudaPersistentDagTask",
        "sizeof_bytes": ctypes.sizeof(CudaPersistentDagTask),
        "field_offsets": {
            "c": CudaPersistentDagTask.c.offset,
            "d": CudaPersistentDagTask.d.offset,
            "tensor_args": CudaPersistentDagTask.tensor_args.offset,
        },
        "kv_pointer_fields": KV_FIELDS,
        "weight_pointer_field": "tensor_args",
    }


def cache_pointer(
    *,
    plan: dict[str, Any],
    cache_name: str,
    field: str,
    ptr: int,
) -> dict[str, Any]:
    return {
        "workload_id": plan["workload_id"],
        "batch_size": plan["batch_size"],
        "cache": cache_name,
        "field": field,
        "device_ptr": ptr,
        "device_ptr_hex": f"0x{ptr:x}",
        "byte_count": plan["bytes"] // 2,
        "layout": plan["layout"],
        "element_dtype": plan["element_dtype"],
    }


def kv_binding_records(
    *,
    lifecycle_plan: dict[str, Any],
    pointer_base: int,
    pointer_stride: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bindings = []
    pointers = []
    index = 0
    for workload in lifecycle_plan.get("workload_plans", []):
        for plan in workload.get("kv_cache_plans", []):
            key_ptr = pointer_base + index * pointer_stride
            value_ptr = key_ptr + pointer_stride // 2
            key = cache_pointer(
                plan=plan,
                cache_name="key_cache",
                field="c",
                ptr=key_ptr,
            )
            value = cache_pointer(
                plan=plan,
                cache_name="value_cache",
                field="d",
                ptr=value_ptr,
            )
            pointers.extend([key, value])
            bindings.append(
                {
                    "workload_id": plan["workload_id"],
                    "batch_size": plan["batch_size"],
                    "status": "ready",
                    "sequence_capacity_tokens": plan["sequence_capacity_tokens"],
                    "prompt_tokens": plan["prompt_tokens"],
                    "decode_tokens": plan["decode_tokens"],
                    "token_position_lifecycle": plan["token_position_lifecycle"],
                    "key_cache": key,
                    "value_cache": value,
                }
            )
            index += 1
    return bindings, pointers


def build_kv_cache_lifecycle(
    *,
    pointer_base: int = 0x70000000,
    pointer_stride: int = 0x10000000,
) -> dict[str, Any]:
    lifecycle_plan = load_lifecycle_plan()
    bindings, pointers = kv_binding_records(
        lifecycle_plan=lifecycle_plan,
        pointer_base=pointer_base,
        pointer_stride=pointer_stride,
    )
    pointer_table = {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_kv_cache_pointer_table",
        "status": "kv_cache_pointer_table_ready",
        "mode": "dry_run_pointer_lifecycle",
        "pointers": pointers,
        "pointer_count": len(pointers),
    }
    closed = {
        **pointer_table,
        "status": "kv_cache_pointer_table_closed",
        "freed_pointer_count": len(pointers),
    }
    return {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_kv_cache_lifecycle",
        "status": "kv_cache_lifecycle_ready",
        "mode": "dry_run_pointer_lifecycle",
        "lifecycle_plan_status": lifecycle_plan.get("status"),
        "abi": dag_task_abi(),
        "pointer_table": pointer_table,
        "closed_pointer_table": closed,
        "kv_cache_bindings": bindings,
        "pointer_count": len(pointers),
        "total_byte_count": sum(item["byte_count"] for item in pointers),
        "implemented_contracts": [
            "kv_cache_key_value_field_binding",
            "kv_cache_token_position_lifecycle",
            "dry_run_pointer_lifecycle",
        ],
        "remaining_runtime_gaps": [
            "cuda_live_kv_cache_owner_in_decode_loop",
            "numerically_correct_qwen_attention_kv_cache_use",
            "decode_loop_execution",
        ],
    }
