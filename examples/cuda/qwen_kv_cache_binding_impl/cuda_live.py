"""CUDA-backed KV-cache pointer owner."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def live_pointer_table(
    *,
    pointers: list[dict[str, Any]],
    host_runtime: Path,
    device: int,
) -> tuple[dict[str, Any], Callable[[], int]]:
    if not host_runtime.is_file():
        return fail_table("host_runtime_missing", host_runtime=host_runtime), lambda: 0

    runtime = load_cuda_runtime(host_runtime)
    ctx = runtime.create_device_context()
    if not ctx:
        return fail_table("create_device_context_failed"), lambda: 0
    init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
    if init_status != 0:
        runtime.destroy_device_context(ctx)
        return fail_table("simpler_init_failed", return_code=init_status), lambda: 0

    ptrs: list[int] = []
    try:
        for pointer in pointers:
            ptr = runtime.device_malloc_ctx(ctx, int(pointer["byte_count"]))
            if not ptr:
                raise RuntimeError(
                    "device allocation failed for "
                    f"{pointer['workload_id']}:{pointer['cache']}"
                )
            ptr_value = int(ptr)
            ptrs.append(ptr_value)
            pointer["device_ptr"] = ptr_value
            pointer["device_ptr_hex"] = f"0x{ptr_value:x}"
    except Exception as exc:  # noqa: BLE001 - artifact records runtime failure.
        cleanup_live(runtime=runtime, ctx=ctx, ptrs=ptrs)
        return fail_table("exception", message=str(exc)), lambda: 0

    def close() -> int:
        return cleanup_live(runtime=runtime, ctx=ctx, ptrs=ptrs)

    table = {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_kv_cache_pointer_table",
        "status": "kv_cache_pointer_table_ready",
        "mode": "cuda_live",
        "device": device,
        "host_runtime": repo_relative(host_runtime),
        "allocation_policy": "allocate_full_kv_cache_without_prefill_copy",
        "pointers": pointers,
        "pointer_count": len(pointers),
        "total_byte_count": sum(item["byte_count"] for item in pointers),
    }
    return table, close


def load_cuda_runtime(host_runtime: Path) -> Any:
    runtime = ctypes.CDLL(str(host_runtime))
    runtime.create_device_context.restype = ctypes.c_void_p
    runtime.destroy_device_context.argtypes = [ctypes.c_void_p]
    runtime.simpler_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.simpler_init.restype = ctypes.c_int
    runtime.finalize_device.argtypes = [ctypes.c_void_p]
    runtime.finalize_device.restype = ctypes.c_int
    runtime.device_malloc_ctx.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    runtime.device_malloc_ctx.restype = ctypes.c_void_p
    runtime.device_free_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return runtime


def fail_table(reason: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_kv_cache_pointer_table",
        "status": "kv_cache_pointer_table_unavailable",
        "reason": reason,
    }
    if "host_runtime" in extra and isinstance(extra["host_runtime"], Path):
        extra["host_runtime"] = repo_relative(extra["host_runtime"])
    return {**payload, **extra}


def cleanup_live(*, runtime: Any, ctx: Any, ptrs: list[int]) -> int:
    freed = 0
    for ptr in ptrs:
        runtime.device_free_ctx(ctx, ctypes.c_void_p(ptr))
        freed += 1
    runtime.finalize_device(ctx)
    runtime.destroy_device_context(ctx)
    return freed
