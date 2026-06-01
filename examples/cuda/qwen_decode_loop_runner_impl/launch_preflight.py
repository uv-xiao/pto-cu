"""Preflight Qwen resource-backed CUDA launch packets."""

from __future__ import annotations

import ctypes
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagArgs,
    CudaPersistentDagState,
    CudaPersistentDagTask,
)

from qwen_decode_loop_runner_impl.launch_helpers import (
    attach_decode_feedback_tensors,
    input_ptr_for_task,
    launch_blockers,
    missing_launch_buffers,
    next_power_of_two,
    output_ptr_for_task,
    parse_ptr,
    set_decode_step_index,
    task_n_for_record,
    task_scalar_arg_count,
    task_scalar_args,
    tensor_arg_count,
    tensor_arg_index,
)
from qwen_decode_loop_runner_impl.submission import QWEN_TASK_FUNCTIONS


CALLABLE_FUNC_IDS = {
    item["callable"]: item["func_id"] for item in QWEN_TASK_FUNCTIONS
}


def launch_packet_preflight(
    *,
    plan: dict[str, Any],
    descriptors: list[dict[str, Any]],
    activation_workspace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_fields = keyed_fields(plan.get("token_pointer_fields", []))
    kv_fields = plan.get("kv_pointer_fields", {})
    workspace = workspace_for_workload(
        activation_workspace=activation_workspace,
        workload_id=plan["workload_id"],
        task_count=len(descriptors),
    )
    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields=kv_fields,
        workspace=workspace,
    )
    workspace_ready = workspace is not None
    return {
        "status": (
            "resource_backed_launch_packet_workspace_bound"
            if packet is not None and workspace_ready
            else "resource_backed_launch_packet_preflight_ready"
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
        "workspace_pointer_policy": workspace_pointer_policy(
            workspace=workspace,
            task_count=len(descriptors),
        ),
        "missing_runtime_buffers": missing_launch_buffers(
            descriptors=descriptors,
            workspace_ready=workspace_ready,
        ),
        "launch_blockers": launch_blockers(workspace_ready=workspace_ready),
        "remaining_gap": (
            "run_prepared_resource_backed_decode_loop"
            if workspace_ready
            else "allocate_safe_qwen_activation_and_logits_buffers"
        ),
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
    workspace: dict[str, Any] | None = None,
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
                workspace=workspace,
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
    workspace: dict[str, Any] | None,
) -> CudaPersistentDagTask:
    tensor_args_t = ctypes.c_void_p * 4
    scalar_args_t = ctypes.c_float * 4
    tensor_args = [0, 0, 0, 0]
    scalar_args = task_scalar_args(
        index=index,
        task_count=task_count,
        descriptor=descriptor,
        workspace=workspace,
    )
    for arg in descriptor.get("tensor_args", [])[:4]:
        tensor_args[tensor_arg_index(arg["arg"])] = parse_ptr(
            arg.get("device_ptr_hex"),
        )
    attach_decode_feedback_tensors(
        index=index,
        task_count=task_count,
        descriptor=descriptor,
        tensor_args=tensor_args,
        token_fields=token_fields,
        workspace=workspace,
    )
    return CudaPersistentDagTask(
        func_id=CALLABLE_FUNC_IDS[descriptor["callable"]],
        a=input_ptr_for_task(
            index=index,
            token_fields=token_fields,
            workspace=workspace,
        ),
        b=parse_ptr(token_fields["b"].get("device_ptr_hex")),
        out=output_ptr_for_task(
            index=index,
            task_count=task_count,
            token_fields=token_fields,
            workspace=workspace,
        ),
        n=task_n_for_record(
            index=index,
            task_count=task_count,
            descriptor=descriptor,
            workspace=workspace,
        ),
        dependent_begin=index,
        dependent_count=1 if index + 1 < task_count else 0,
        initial_fanin=0 if index == 0 else 1,
        c=parse_ptr(kv_fields["c"].get("device_ptr_hex")),
        d=parse_ptr(kv_fields["d"].get("device_ptr_hex")),
        tensor_args=tensor_args_t(*tensor_args),
        scalar_args=scalar_args_t(*scalar_args),
        tensor_arg_count=tensor_arg_count(tensor_args),
        scalar_arg_count=task_scalar_arg_count(scalar_args),
    )


def workspace_for_workload(
    *,
    activation_workspace: dict[str, Any] | None,
    workload_id: str,
    task_count: int,
) -> dict[str, Any] | None:
    if activation_workspace is None:
        return None
    if activation_workspace.get("status") != "activation_workspace_lifecycle_ready":
        return None
    pointer_table = activation_workspace.get("pointer_table", {})
    if pointer_table.get("mode") != "cuda_live":
        return None
    for item in pointer_table.get("pointer_sets", []):
        if item.get("workload_id") != workload_id:
            continue
        if len(item.get("activation_buffers", [])) < max(task_count - 1, 0):
            return None
        if not item.get("logits_buffer", {}).get("device_ptr_hex"):
            return None
        return item
    return None


def workspace_pointer_policy(
    *,
    workspace: dict[str, Any] | None,
    task_count: int,
) -> dict[str, Any]:
    if workspace is None:
        return {
            "status": "not_bound",
            "activation_buffer_count": 0,
            "logits_buffer": "not_bound",
        }
    return {
        "status": "workspace_bound",
        "task_input_chain": "task0_uses_token_ids_then_activation_i_minus_1",
        "task_output_chain": "intermediate_tasks_use_activation_i_last_uses_logits",
        "activation_buffer_count": len(workspace.get("activation_buffers", [])),
        "required_activation_buffer_count": max(task_count - 1, 0),
        "logits_buffer": workspace["logits_buffer"]["device_ptr_hex"],
        "total_byte_count": workspace["total_byte_count"],
    }
