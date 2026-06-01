"""Preflight Qwen resource-backed CUDA launch packets."""

from __future__ import annotations

import ctypes
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagArgs,
    CudaPersistentDagState,
    CudaPersistentDagTask,
)

from qwen_decode_loop_runner_impl.submission import QWEN_TASK_FUNCTIONS


CALLABLE_FUNC_IDS = {
    item["callable"]: item["func_id"] for item in QWEN_TASK_FUNCTIONS
}


def launch_packet_preflight(
    *,
    plan: dict[str, Any],
    descriptors: list[dict[str, Any]],
) -> dict[str, Any]:
    token_fields = keyed_fields(plan.get("token_pointer_fields", []))
    kv_fields = plan.get("kv_pointer_fields", {})
    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields=kv_fields,
    )
    return {
        "status": (
            "resource_backed_launch_packet_preflight_ready"
            if packet is not None
            else "resource_backed_launch_packet_preflight_incomplete"
        ),
        "execution_status": "not_launched",
        "task_struct_size_bytes": ctypes.sizeof(CudaPersistentDagTask),
        "state_struct_size_bytes": ctypes.sizeof(CudaPersistentDagState),
        "dag_args_size_bytes": ctypes.sizeof(CudaPersistentDagArgs),
        "host_task_packet_bytes": (
            ctypes.sizeof(packet) if packet is not None else 0
        ),
        "task_count": len(descriptors),
        "dependent_count": max(len(descriptors) - 1, 0),
        "queue_capacity": next_power_of_two(max(len(descriptors) + 1, 16)),
        "scheduler_blocks": 1,
        "worker_blocks": 1,
        "block_dim": 64,
        "token_pointer_fields": sorted(token_fields),
        "kv_pointer_fields": sorted(kv_fields),
        "missing_runtime_buffers": missing_launch_buffers(descriptors),
        "launch_blockers": [
            "intermediate_activation_buffers_not_allocated",
            "logits_output_dtype_mismatch_with_output_ids",
            "diagnostic_kernel_bodies_not_full_qwen_numeric",
        ],
        "remaining_gap": "allocate_safe_qwen_activation_and_logits_buffers",
    }


def keyed_fields(bindings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item["field"]: item
        for item in bindings
        if isinstance(item, dict) and item.get("device_ptr_hex")
    }


def build_host_task_packet(
    *,
    descriptors: list[dict[str, Any]],
    token_fields: dict[str, dict[str, Any]],
    kv_fields: dict[str, Any],
) -> Any | None:
    if not descriptors or {"a", "b", "out"} - set(token_fields):
        return None
    if {"c", "d"} - set(kv_fields):
        return None
    task_t = CudaPersistentDagTask * len(descriptors)
    return task_t(
        *[
            host_task_record(
                index=index,
                task_count=len(descriptors),
                descriptor=descriptor,
                token_fields=token_fields,
                kv_fields=kv_fields,
            )
            for index, descriptor in enumerate(descriptors)
        ]
    )


def host_task_record(
    *,
    index: int,
    task_count: int,
    descriptor: dict[str, Any],
    token_fields: dict[str, dict[str, Any]],
    kv_fields: dict[str, Any],
) -> CudaPersistentDagTask:
    tensor_args_t = ctypes.c_void_p * 4
    scalar_args_t = ctypes.c_float * 4
    tensor_args = [0, 0, 0, 0]
    for arg in descriptor.get("tensor_args", [])[:4]:
        tensor_args[tensor_arg_index(arg["arg"])] = parse_ptr(
            arg.get("device_ptr_hex"),
        )
    return CudaPersistentDagTask(
        func_id=CALLABLE_FUNC_IDS[descriptor["callable"]],
        a=parse_ptr(token_fields["a"].get("device_ptr_hex")),
        b=parse_ptr(token_fields["b"].get("device_ptr_hex")),
        out=parse_ptr(token_fields["out"].get("device_ptr_hex")),
        n=1,
        dependent_begin=index,
        dependent_count=1 if index + 1 < task_count else 0,
        initial_fanin=0 if index == 0 else 1,
        c=parse_ptr(kv_fields["c"].get("device_ptr_hex")),
        d=parse_ptr(kv_fields["d"].get("device_ptr_hex")),
        tensor_args=tensor_args_t(*tensor_args),
        scalar_args=scalar_args_t(0.0, 0.0, 0.0, 0.0),
        tensor_arg_count=min(len(descriptor.get("tensor_args", [])), 4),
        scalar_arg_count=0,
    )


def tensor_arg_index(value: str) -> int:
    prefix = "tensor_args["
    if not value.startswith(prefix) or not value.endswith("]"):
        return 0
    parsed = int(value[len(prefix) : -1])
    return parsed if 0 <= parsed < 4 else 0


def parse_ptr(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        return int(value, 0)
    return 0


def missing_launch_buffers(descriptors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "buffer": "intermediate_activation_buffers",
            "required_count": max(len(descriptors) - 1, 0),
            "status": "not_allocated",
        },
        {
            "buffer": "float_logits_or_sampling_output",
            "required_count": 1,
            "status": "not_allocated",
        },
    ]


def next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power
