"""Build Qwen activation workspace pointer tables."""

from __future__ import annotations

import ctypes
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
    return pointer_set(plan=plan, activation_buffers=activation_buffers, logits=logits), (
        next_ptr + pointer_stride
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
    return pointer_set(plan=plan, activation_buffers=activation_buffers, logits=logits)


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


def pointer_set(
    *,
    plan: dict[str, Any],
    activation_buffers: list[dict[str, Any]],
    logits: dict[str, Any],
) -> dict[str, Any]:
    return {
        "workload_id": plan["workload_id"],
        "status": "workspace_pointers_ready",
        "pointer_count": len(activation_buffers) + 1,
        "activation_buffer_count": len(activation_buffers),
        "activation_buffers": activation_buffers,
        "logits_buffer": logits,
        "total_byte_count": plan["total_byte_count"],
    }


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
