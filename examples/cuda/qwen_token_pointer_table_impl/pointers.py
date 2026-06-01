"""Build dry-run or live CUDA token pointer tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from qwen_token_pointer_table_impl.common import (
    ROOT,
    TOKEN_BUFFER_SCRIPT,
    TOKEN_BUFFERS,
    load_module,
    repo_relative,
)


def pointer_record(
    *,
    record: dict[str, Any],
    buffer: str,
    ptr: int,
) -> dict[str, Any]:
    desc = record[f"{buffer}_device_buffer"]
    return {
        "workload_id": record["workload_id"],
        "buffer": buffer,
        "device_ptr": ptr,
        "device_ptr_hex": f"0x{ptr:x}",
        "byte_count": desc["byte_count"],
        "shape": desc["shape"],
        "dtype": desc["dtype"],
    }


def build_token_binding(
    *,
    mode: str,
    cache_dir: Path | None,
) -> dict[str, Any]:
    module = load_module(TOKEN_BUFFER_SCRIPT, "qwen_token_buffer_for_pointer_table")
    return module.build_cuda_token_buffer_binding(
        mode=mode,
        cache_dir=cache_dir,
        no_cuda_probe=True,
    )


def dry_run_pointer_table(
    *,
    token_binding: dict[str, Any],
    pointer_base: int,
    pointer_stride: int,
) -> dict[str, Any]:
    pointers = []
    for index, record in enumerate(token_binding.get("workload_records", [])):
        for buffer_index, buffer in enumerate(TOKEN_BUFFERS):
            ptr = pointer_base + (index * len(TOKEN_BUFFERS) + buffer_index) * (
                pointer_stride
            )
            pointers.append(pointer_record(record=record, buffer=buffer, ptr=ptr))
    return ready_table(mode="dry_run_pointer_lifecycle", pointers=pointers)


def live_pointer_table(
    *,
    token_binding: dict[str, Any],
    mode: str,
    cache_dir: Path | None,
    host_runtime: Path,
    device: int,
) -> tuple[dict[str, Any], Callable[[], int]]:
    token_module = load_module(TOKEN_BUFFER_SCRIPT, "qwen_token_buffer_for_live_ptrs")
    runtime_binding = token_module.load_runtime_input_binding(
        mode=mode,
        cache_dir=cache_dir,
    )
    if runtime_binding.get("status") != "runtime_input_binding_plan_ready":
        return fail_table("runtime_input_binding_unavailable"), lambda: 0
    if not host_runtime.is_file():
        return fail_table("host_runtime_missing", host_runtime=host_runtime), lambda: 0

    weight_module = load_module(
        ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py",
        "qwen_cuda_weight_binding_for_token_ptrs",
    )
    runtime = weight_module.load_cuda_runtime(host_runtime)
    ctx = runtime.create_device_context()
    if not ctx:
        return fail_table("create_device_context_failed"), lambda: 0
    init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
    if init_status != 0:
        runtime.destroy_device_context(ctx)
        return fail_table("simpler_init_failed", return_code=init_status), lambda: 0

    records = {
        item["workload_id"]: item
        for item in token_binding.get("workload_records", [])
    }
    ptrs: list[int] = []
    pointers: list[dict[str, Any]] = []
    try:
        for runtime_record in runtime_binding.get("workload_records", []):
            record = records[runtime_record["workload_id"]]
            for buffer, host_bytes in token_module.record_host_buffers(
                runtime_record
            ).items():
                ptr = token_module.copy_and_verify_buffer(
                    runtime=runtime,
                    ctx=ctx,
                    host_bytes=host_bytes,
                )
                ptrs.append(ptr)
                pointers.append(pointer_record(record=record, buffer=buffer, ptr=ptr))
    except Exception as exc:  # noqa: BLE001 - artifact records runtime failure.
        cleanup_live(runtime=runtime, ctx=ctx, ptrs=ptrs)
        return fail_table("exception", message=str(exc)), lambda: 0

    def close() -> int:
        return cleanup_live(runtime=runtime, ctx=ctx, ptrs=ptrs)

    table = ready_table(mode="cuda_live", pointers=pointers)
    table.update({"device": device, "host_runtime": repo_relative(host_runtime)})
    return table, close


def ready_table(*, mode: str, pointers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_token_pointer_table",
        "status": "cuda_token_pointer_table_ready",
        "mode": mode,
        "pointers": pointers,
        "pointer_count": len(pointers),
    }


def fail_table(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_token_pointer_table",
        "status": "cuda_token_pointer_table_unavailable",
        "reason": reason,
    }
    if "host_runtime" in extra and isinstance(extra["host_runtime"], Path):
        extra["host_runtime"] = repo_relative(extra["host_runtime"])
    return {**payload, **extra}


def cleanup_live(*, runtime: Any, ctx: Any, ptrs: list[int]) -> int:
    freed = 0
    for ptr in ptrs:
        runtime.device_free_ctx(ctx, ptr)
        freed += 1
    runtime.finalize_device(ctx)
    runtime.destroy_device_context(ctx)
    return freed
