"""Single-context resource allocators for Qwen runner sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import qwen_cuda_token_buffer_binding as token_buffers
import qwen_cuda_weight_binding as weight_runtime
from qwen_decode_loop_runner_impl.activation_workspace import (
    load_model_shape,
    workspace_plan,
)
from qwen_decode_loop_runner_impl.workspace_pointers import allocate_pointer_set
from qwen_kv_cache_binding_impl.lifecycle import kv_binding_records, load_lifecycle_plan
from qwen_kv_cache_binding_impl.zeroing import (
    KV_ZERO_CHUNK_BYTES,
    zero_device_allocation,
)
from qwen_resident_weight_table_impl.common import (
    DEFAULT_COPY_CHUNK_BYTES,
    load_or_build_weight_binding,
    load_weight_args_path,
    repo_relative,
)
from qwen_resident_weight_table_impl.materialization import (
    materialize_with_pointer_table,
)
from qwen_resident_weight_table_impl.resident import pointer_record
from qwen_token_pointer_table_impl.lifecycle import materialize_decode_args
from qwen_token_pointer_table_impl.pointers import (
    build_token_binding,
    pointer_record as token_pointer_record,
    ready_table as token_ready_table,
)


class GroupedAllocationList:
    def __init__(self, target: list[tuple[str, int]], group: str) -> None:
        self.target = target
        self.group = group

    def append(self, ptr: int) -> None:
        self.target.append((self.group, int(ptr)))


def selected_workload_ids(workload_ids: list[str] | None) -> set[str] | None:
    if not workload_ids:
        return None
    return {str(item) for item in workload_ids}


def selected_batch_size(batch_size: int | None) -> int | None:
    if batch_size is None:
        return None
    selected = int(batch_size)
    if selected <= 0:
        raise ValueError("batch_size must be positive")
    return selected


def resized_buffer_descriptor(
    descriptor: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    shape = [int(item) for item in descriptor.get("shape", [])]
    if not shape:
        return descriptor
    resized_shape = [batch_size, *shape[1:]]
    element_count = 1
    for dim in resized_shape:
        element_count *= dim
    return {
        **descriptor,
        "shape": resized_shape,
        "element_count": element_count,
        "byte_count": element_count * 4,
    }


def resized_workload_record(
    record: dict[str, Any],
    *,
    batch_size: int,
) -> dict[str, Any]:
    batch_sizes = record.get("batch_sizes")
    if (
        isinstance(batch_sizes, list)
        and batch_size not in [int(item) for item in batch_sizes]
    ):
        raise ValueError(
            f"batch_size {batch_size} is not in workload "
            f"{record.get('workload_id')} batch_sizes"
        )
    resized = {
        **record,
        "batch_sizes": [batch_size],
        "scalar_bindings": {
            **record.get("scalar_bindings", {}),
            "max_batch_size": batch_size,
        },
    }
    for name in ("input_ids", "attention_mask", "output_ids"):
        for suffix in ("_buffer", "_device_buffer"):
            key = f"{name}{suffix}"
            if key in resized:
                resized[key] = resized_buffer_descriptor(
                    resized[key],
                    batch_size=batch_size,
                )
    return resized


def filter_workload_records(
    payload: dict[str, Any],
    *,
    workload_ids: list[str] | None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    selected = selected_workload_ids(workload_ids)
    batch = selected_batch_size(batch_size)
    if selected is None and batch is None:
        return payload
    records = payload.get("workload_records", [])
    if selected is not None:
        records = [
            item
            for item in records
            if item.get("workload_id") in selected
        ]
    if batch is not None:
        records = [
            resized_workload_record(item, batch_size=batch)
            for item in records
        ]
    return {
        **payload,
        "workload_records": records,
    }


def open_token_table(
    *,
    runtime: Any,
    ctx: Any,
    mode: str,
    cache_dir: Path | None,
    host_runtime: Path,
    device: int,
    allocations: list[tuple[str, int]],
    workload_ids: list[str] | None = None,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    token_binding = filter_workload_records(
        build_token_binding(mode=mode, cache_dir=cache_dir),
        workload_ids=workload_ids,
        batch_size=batch_size,
    )
    runtime_binding = filter_workload_records(
        token_buffers.load_runtime_input_binding(
            mode=mode,
            cache_dir=cache_dir,
        ),
        workload_ids=workload_ids,
        batch_size=batch_size,
    )
    records = {
        item["workload_id"]: item for item in token_binding.get("workload_records", [])
    }
    pointers = []
    for runtime_record in runtime_binding.get("workload_records", []):
        record = records[runtime_record["workload_id"]]
        for buffer, host_bytes in token_buffers.record_host_buffers(
            runtime_record,
        ).items():
            ptr = token_buffers.copy_and_verify_buffer(
                runtime=runtime,
                ctx=ctx,
                host_bytes=host_bytes,
            )
            allocations.append(("token_pointer_table", ptr))
            pointers.append(token_pointer_record(record=record, buffer=buffer, ptr=ptr))
    token_table = token_ready_table(mode="cuda_live", pointers=pointers)
    token_table.update(
        {
            "device": device,
            "host_runtime": repo_relative(host_runtime),
            "cuda_context_policy": "single_context_session",
        }
    )
    decode_args = materialize_decode_args(
        token_binding=token_binding,
        pointer_table=token_table,
    )
    return token_binding, token_table, decode_args


def open_kv_table(
    *,
    runtime: Any,
    ctx: Any,
    host_runtime: Path,
    device: int,
    allocations: list[tuple[str, int]],
    workload_ids: list[str] | None = None,
    batch_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kv_bindings, pointers = kv_binding_records(
        lifecycle_plan=load_lifecycle_plan(),
        pointer_base=0,
        pointer_stride=0,
    )
    selected = selected_workload_ids(workload_ids)
    batch = selected_batch_size(batch_size)
    if selected is not None:
        kv_bindings = [
            item for item in kv_bindings if item.get("workload_id") in selected
        ]
        pointers = [
            item for item in pointers if item.get("workload_id") in selected
        ]
    if batch is not None:
        kv_bindings = [
            item for item in kv_bindings if int(item.get("batch_size", 0)) == batch
        ]
        pointers = [
            item for item in pointers if int(item.get("batch_size", 0)) == batch
        ]
    for pointer in pointers:
        ptr = runtime.device_malloc_ctx(ctx, int(pointer["byte_count"]))
        if not ptr:
            raise RuntimeError(
                "device allocation failed for "
                f"{pointer['workload_id']}:{pointer['cache']}"
            )
        ptr_value = int(ptr)
        zero_device_allocation(
            runtime=runtime,
            ctx=ctx,
            ptr=ptr_value,
            byte_count=int(pointer["byte_count"]),
        )
        allocations.append(("kv_cache", ptr_value))
        pointer["device_ptr"] = ptr_value
        pointer["device_ptr_hex"] = f"0x{ptr_value:x}"
    return kv_bindings, {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_kv_cache_pointer_table",
        "status": "kv_cache_pointer_table_ready",
        "mode": "cuda_live",
        "device": device,
        "host_runtime": repo_relative(host_runtime),
        "cuda_context_policy": "single_context_session",
        "allocation_policy": "allocate_zeroed_full_kv_cache_without_prefill_copy",
        "initialization_policy": {
            "state": "zero_initialized",
            "chunk_bytes": KV_ZERO_CHUNK_BYTES,
            "scope": "entire_key_value_cache_allocation",
        },
        "pointers": pointers,
        "pointer_count": len(pointers),
        "total_byte_count": sum(item["byte_count"] for item in pointers),
    }


def open_resident_table(
    *,
    runtime: Any,
    ctx: Any,
    host_runtime: Path,
    device: int,
    allocations: list[tuple[str, int]],
) -> tuple[str, Path | None, dict[str, Any], dict[str, Any]]:
    weight_binding, binding_source = load_or_build_weight_binding(None)
    bindings = weight_binding.get("bindings", [])
    if not isinstance(bindings, list):
        raise ValueError("weight binding artifact has no bindings list")
    pointers = []
    for item in bindings:
        ptr = runtime.device_malloc_ctx(ctx, int(item["size_bytes"]))
        if not ptr:
            raise RuntimeError(f"device allocation failed for {item['tensor']}")
        ptr_value = int(ptr)
        allocations.append(("resident_weight_table", ptr_value))
        weight_runtime.copy_file_range_to_device(
            runtime=runtime,
            ctx=ctx,
            binding=item,
            dev_ptr=ptr_value,
            chunk_bytes=DEFAULT_COPY_CHUNK_BYTES,
        )
        pointers.append(pointer_record(item, ptr_value))
    resident_table = {
        "schema_version": 1,
        "kind": "pto_qwen_resident_weight_pointer_table",
        "status": "resident_weight_pointer_table_ready",
        "model_id": weight_binding["model_id"],
        "model_revision": weight_binding["model_revision"],
        "device": device,
        "source": repo_relative(host_runtime),
        "lifetime": "valid_until_session_close",
        "cuda_context_policy": "single_context_session",
        "pointer_count": len(pointers),
        "freed_pointer_count": 0,
        "resident_bytes": sum(int(item["size_bytes"]) for item in pointers),
        "pointers": pointers,
    }
    weight_args_path = load_weight_args_path(None)
    materialization = (
        materialize_with_pointer_table(
            weight_args_json=weight_args_path,
            weight_binding_json=None,
            pointer_table=resident_table,
        )
        if weight_args_path is not None
        else {}
    )
    return binding_source, weight_args_path, resident_table, materialization


def open_workspace(
    *,
    runtime: Any,
    ctx: Any,
    host_runtime: Path,
    device: int,
    plans: list[dict[str, Any]],
    graph_task_count: int,
    allocations: list[tuple[str, int]],
    descriptors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model_shape = load_model_shape()
    workspace_plans = [
        workspace_plan(
            plan=plan,
            graph_task_count=graph_task_count,
            model_shape=model_shape,
            descriptors=descriptors,
        )
        for plan in plans
    ]
    pointer_sets = [
        allocate_pointer_set(
            runtime,
            ctx,
            plan,
            GroupedAllocationList(allocations, "activation_workspace"),
        )
        for plan in workspace_plans
    ]
    pointer_count = sum(item["pointer_count"] for item in pointer_sets)
    return {
        "schema_version": 1,
        "kind": "pto_qwen_activation_workspace_lifecycle",
        "status": "activation_workspace_lifecycle_ready",
        "mode": "cuda_live",
        "runtime": "cuda/persistent_device",
        "model_id": model_shape["model_id"],
        "hidden_size": model_shape["hidden_size"],
        "vocab_size": model_shape["vocab_size"],
        "element_dtype": "float32",
        "workspace_plans": workspace_plans,
        "pointer_table": {
            "schema_version": 1,
            "kind": "pto_qwen_activation_workspace_pointer_table",
            "status": "activation_workspace_pointer_table_ready",
            "mode": "cuda_live",
            "device": device,
            "host_runtime": repo_relative(host_runtime),
            "cuda_context_policy": "single_context_session",
            "pointer_sets": pointer_sets,
            "pointer_count": pointer_count,
            "freed_pointer_count": 0,
        },
        "implemented_contracts": [
            "activation_workspace_lifetime_owner",
            "float_logits_or_sampling_output_workspace",
            "cuda_live_activation_workspace",
            "single_context_session_workspace",
        ],
        "remaining_runtime_gaps": [
            "attach_workspace_to_run_prepared_state",
            "numerically_correct_qwen_kernel_bodies",
            "cuda_live_decode_loop_execution",
        ],
    }
