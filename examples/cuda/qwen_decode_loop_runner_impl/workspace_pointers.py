"""Build Qwen activation workspace pointer tables."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any

from qwen_persistent_proxy_live_impl.runtime import bind_persistent_runtime


ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def dry_run_workspace_table(
    *,
    workspace_plans: list[dict[str, Any]],
    pointer_base: int = 0xA0000000,
    pointer_stride: int = 0x200000,
) -> dict[str, Any]:
    pointer_sets = []
    next_ptr = pointer_base
    for plan in workspace_plans:
        pointer_set, next_ptr = pointer_set_for_plan(
            plan=plan,
            base_ptr=next_ptr,
            pointer_stride=pointer_stride,
        )
        pointer_sets.append(pointer_set)
    pointer_count = sum(item["pointer_count"] for item in pointer_sets)
    return {
        "schema_version": 1,
        "kind": "pto_qwen_activation_workspace_pointer_table",
        "status": "activation_workspace_pointer_table_ready",
        "mode": "dry_run_pointer_lifecycle",
        "pointer_sets": pointer_sets,
        "pointer_count": pointer_count,
        "freed_pointer_count": pointer_count,
    }


def live_workspace_table(
    *,
    workspace_plans: list[dict[str, Any]],
    device: int,
    host_runtime: Path,
) -> dict[str, Any]:
    if not host_runtime.is_file():
        return unavailable_workspace_table("host_runtime_missing", host_runtime)
    runtime = ctypes.CDLL(str(host_runtime))
    bind_persistent_runtime(runtime)
    ctx = runtime.create_device_context()
    if not ctx:
        return unavailable_workspace_table("create_device_context_failed", host_runtime)
    init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
    if init_status != 0:
        runtime.destroy_device_context(ctx)
        return unavailable_workspace_table(
            "simpler_init_failed",
            host_runtime,
            return_code=init_status,
        )
    allocated: list[int] = []
    try:
        pointer_sets = [
            allocate_pointer_set(runtime, ctx, plan, allocated)
            for plan in workspace_plans
        ]
    except Exception as exc:  # noqa: BLE001 - preserve runtime failure evidence.
        freed = cleanup(runtime, ctx, allocated)
        return unavailable_workspace_table(
            "allocation_failed",
            host_runtime,
            message=str(exc),
            freed_pointer_count=freed,
        )
    freed = cleanup(runtime, ctx, allocated)
    return {
        "schema_version": 1,
        "kind": "pto_qwen_activation_workspace_pointer_table",
        "status": "activation_workspace_pointer_table_ready",
        "mode": "cuda_live",
        "device": device,
        "host_runtime": repo_relative(host_runtime),
        "pointer_sets": pointer_sets,
        "pointer_count": len(allocated),
        "freed_pointer_count": freed,
    }


def pointer_set_for_plan(
    *,
    plan: dict[str, Any],
    base_ptr: int,
    pointer_stride: int,
) -> tuple[dict[str, Any], int]:
    activation_buffers = []
    next_ptr = base_ptr
    for index in range(plan["activation_buffer_count"]):
        element_count = activation_buffer_elements(plan, index)
        activation_buffers.append(
            buffer_record(
                name=f"activation_{index}",
                role="intermediate_activation",
                ptr=next_ptr,
                byte_count=element_count * 4,
                element_count=element_count,
            )
        )
        next_ptr += pointer_stride
    logits = buffer_record(
        name="logits",
        role="float_logits_or_sampling_output",
        ptr=next_ptr,
        byte_count=plan["logits_buffer_bytes"],
        element_count=plan["logits_buffer_elements"],
    )
    next_ptr += pointer_stride
    rope_tables = rope_table_records(
        plan=plan,
        base_ptr=next_ptr,
        stride=pointer_stride,
    )
    return pointer_set(
        plan=plan,
        activation_buffers=activation_buffers,
        logits=logits,
        rope_tables=rope_tables,
    ), (
        next_ptr + pointer_stride * len(rope_tables)
    )


def allocate_pointer_set(
    runtime: Any,
    ctx: Any,
    plan: dict[str, Any],
    allocated: list[int],
) -> dict[str, Any]:
    activation_buffers = [
        allocate_record(
            runtime,
            ctx,
            allocated,
            name=f"activation_{index}",
            role="intermediate_activation",
            byte_count=activation_buffer_elements(plan, index) * 4,
            element_count=activation_buffer_elements(plan, index),
        )
        for index in range(plan["activation_buffer_count"])
    ]
    logits = allocate_record(
        runtime,
        ctx,
        allocated,
        name="logits",
        role="float_logits_or_sampling_output",
        byte_count=plan["logits_buffer_bytes"],
        element_count=plan["logits_buffer_elements"],
    )
    rope_tables = [
        allocate_record(
            runtime,
            ctx,
            allocated,
            name=name,
            role=name,
            byte_count=plan["rope_table_bytes"],
            element_count=plan["rope_table_elements"],
        )
        for name in ("rope_cos_table", "rope_sin_table")
    ]
    initialize_rope_tables(runtime, ctx, rope_tables, plan=plan)
    return pointer_set(
        plan=plan,
        activation_buffers=activation_buffers,
        logits=logits,
        rope_tables=rope_tables,
    )


def allocate_record(
    runtime: Any,
    ctx: Any,
    allocated: list[int],
    *,
    name: str,
    role: str,
    byte_count: int,
    element_count: int,
) -> dict[str, Any]:
    ptr = runtime.device_malloc_ctx(ctx, byte_count)
    if not ptr:
        raise RuntimeError(f"device allocation failed for {name}")
    ptr_value = int(ptr)
    allocated.append(ptr_value)
    return buffer_record(
        name=name,
        role=role,
        ptr=ptr_value,
        byte_count=byte_count,
        element_count=element_count,
    )


def initialize_rope_tables(
    runtime: Any,
    ctx: Any,
    rope_tables: list[dict[str, Any]],
    plan: dict[str, Any],
) -> None:
    tables = rope_table_values(plan)
    for item in rope_tables:
        values = tables[item["role"]]
        host = (ctypes.c_float * len(values))(*values)
        status = runtime.copy_to_device_ctx(
            ctx,
            ctypes.c_void_p(device_ptr_value(item)),
            ctypes.byref(host),
            ctypes.sizeof(host),
        )
        if status != 0:
            raise RuntimeError(f"copy_to_device {item['role']} failed")
        item["initialization"] = plan["rope_table_policy"]
        item["base_position"] = int(plan["rope_base_position"])
        item["rope_theta"] = float(plan["rope_theta"])


def device_ptr_value(item: dict[str, Any]) -> int:
    if item.get("device_ptr"):
        return int(item["device_ptr"])
    return int(str(item["device_ptr_hex"]), 0)


def refresh_rope_tables_for_decode_position(
    runtime: Any,
    ctx: Any,
    workspace: dict[str, Any],
    *,
    decode_position: int,
) -> dict[str, Any]:
    rope_tables = workspace_rope_tables(workspace)
    if len(rope_tables) != 2:
        return {
            "status": "not_refreshed",
            "reason": "runtime_rope_tables_not_bound",
        }
    plan = {
        "rope_base_position": int(decode_position),
        "rope_table_elements": int(rope_tables[0]["element_count"]),
        "rope_table_policy": "position_correct_for_decode_step",
        "rope_theta": float(rope_tables[0].get("rope_theta", 1000000.0)),
    }
    initialize_rope_tables(runtime, ctx, rope_tables, plan=plan)
    return {
        "status": "refreshed",
        "policy": "position_correct_for_decode_step",
        "decode_position": int(decode_position),
        "rope_theta": float(plan["rope_theta"]),
        "rope_table_elements": int(plan["rope_table_elements"]),
        "runtime_buffers": {
            item["role"]: item["device_ptr_hex"] for item in rope_tables
        },
    }


def workspace_rope_tables(workspace: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_buffers = workspace.get("runtime_buffers", {})
    if not isinstance(runtime_buffers, dict):
        return []
    tables = []
    for role in ("rope_cos_table", "rope_sin_table"):
        item = runtime_buffers.get(role)
        if not isinstance(item, dict):
            return []
        tables.append(item)
    return tables


def rope_table_values(plan: dict[str, Any]) -> dict[str, list[float]]:
    count = int(plan["rope_table_elements"])
    head_dim = max(count * 2, 1)
    position = int(plan["rope_base_position"])
    theta = float(plan["rope_theta"])
    angles = [
        position / (theta ** (float(index * 2) / float(head_dim)))
        for index in range(count)
    ]
    return {
        "rope_cos_table": [math.cos(angle) for angle in angles],
        "rope_sin_table": [math.sin(angle) for angle in angles],
    }


def pointer_set(
    *,
    plan: dict[str, Any],
    activation_buffers: list[dict[str, Any]],
    logits: dict[str, Any],
    rope_tables: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_buffers = {item["role"]: item for item in rope_tables}
    return {
        "workload_id": plan["workload_id"],
        "status": "workspace_pointers_ready",
        "pointer_count": len(activation_buffers) + 1 + len(rope_tables),
        "activation_buffer_count": len(activation_buffers),
        "activation_buffers": activation_buffers,
        "logits_buffer": logits,
        "rope_tables": rope_tables,
        "runtime_buffers": runtime_buffers,
        "total_byte_count": plan["total_byte_count"],
    }


def rope_table_records(
    *,
    plan: dict[str, Any],
    base_ptr: int,
    stride: int,
) -> list[dict[str, Any]]:
    records = [
        {
            **buffer_record(
                name=name,
                role=name,
                ptr=base_ptr + index * stride,
                byte_count=plan["rope_table_bytes"],
                element_count=plan["rope_table_elements"],
            ),
            "initialization": plan["rope_table_policy"],
            "base_position": int(plan["rope_base_position"]),
            "rope_theta": float(plan["rope_theta"]),
        }
        for index, name in enumerate(("rope_cos_table", "rope_sin_table"))
    ]
    return records


def buffer_record(
    *,
    name: str,
    role: str,
    ptr: int,
    byte_count: int,
    element_count: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "device_ptr": ptr,
        "device_ptr_hex": f"0x{ptr:x}",
        "byte_count": byte_count,
        "element_count": element_count,
        "element_dtype": "float32",
    }


def activation_buffer_elements(plan: dict[str, Any], index: int) -> int:
    counts = plan.get("activation_buffer_element_counts", [])
    if isinstance(counts, list) and index < len(counts):
        return int(counts[index])
    return int(plan["activation_buffer_elements"])


def unavailable_workspace_table(reason: str, host_runtime: Path, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "pto_qwen_activation_workspace_pointer_table",
        "status": "activation_workspace_pointer_table_unavailable",
        "reason": reason,
        "host_runtime": repo_relative(host_runtime),
        **extra,
    }


def cleanup(runtime: Any, ctx: Any, ptrs: list[int]) -> int:
    freed = 0
    for ptr in reversed(ptrs):
        runtime.device_free_ctx(ctx, ctypes.c_void_p(ptr))
        freed += 1
    runtime.finalize_device(ctx)
    runtime.destroy_device_context(ctx)
    return freed
