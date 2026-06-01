"""CUDA runtime copy and residency probes for Qwen weights."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from .common import ROOT, repo_relative


def read_tensor_bytes(binding: dict[str, Any]) -> bytes:
    path = ROOT / binding["shard_path"]
    if not path.is_file():
        path = Path(binding["shard_path"])
    start, end = binding["file_absolute_offsets"]
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    if len(data) != end - start:
        raise ValueError(f"{path} ended before {binding['tensor']} tensor bytes")
    return data


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
    runtime.copy_to_device_ctx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.copy_to_device_ctx.restype = ctypes.c_int
    runtime.copy_from_device_ctx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.copy_from_device_ctx.restype = ctypes.c_int
    return runtime


def copy_file_range_to_device(
    *,
    runtime: Any,
    ctx: Any,
    binding: dict[str, Any],
    dev_ptr: int,
    chunk_bytes: int,
) -> None:
    path = ROOT / binding["shard_path"]
    if not path.is_file():
        path = Path(binding["shard_path"])
    start, end = binding["file_absolute_offsets"]
    copied = 0
    remaining = end - start
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(chunk_bytes, remaining))
            if not chunk:
                raise ValueError(
                    f"{path} ended before {binding['tensor']} tensor bytes"
                )
            buffer = ctypes.create_string_buffer(chunk, len(chunk))
            status = runtime.copy_to_device_ctx(
                ctx,
                ctypes.c_void_p(dev_ptr + copied),
                ctypes.cast(buffer, ctypes.c_void_p),
                len(chunk),
            )
            if status != 0:
                raise RuntimeError(
                    f"copy_to_device failed for {binding['tensor']} "
                    f"at offset {copied}: {status}"
                )
            copied += len(chunk)
            remaining -= len(chunk)


def verify_device_prefix(
    *,
    runtime: Any,
    ctx: Any,
    binding: dict[str, Any],
    dev_ptr: int,
    verify_bytes: int,
) -> bool:
    expected = read_tensor_bytes(
        {
            **binding,
            "file_absolute_offsets": [
                binding["file_absolute_offsets"][0],
                min(
                    binding["file_absolute_offsets"][1],
                    binding["file_absolute_offsets"][0] + verify_bytes,
                ),
            ],
        }
    )
    output = ctypes.create_string_buffer(len(expected))
    status = runtime.copy_from_device_ctx(
        ctx,
        ctypes.cast(output, ctypes.c_void_p),
        ctypes.c_void_p(dev_ptr),
        len(expected),
    )
    if status != 0:
        raise RuntimeError(
            f"copy_from_device failed for {binding['tensor']}: {status}"
        )
    return output.raw == expected


def run_cuda_copy_probe(
    *,
    bindings: list[dict[str, Any]],
    device: int,
    host_runtime: Path,
    max_tensor_bytes: int,
    max_total_bytes: int,
    max_tensors: int,
) -> dict[str, Any]:
    candidates = [
        item
        for item in bindings
        if item["size_bytes"] <= max_tensor_bytes
    ]
    candidates.sort(key=lambda item: (item["size_bytes"], item["tensor"]))
    selected = []
    total = 0
    for item in candidates:
        if len(selected) >= max_tensors:
            break
        if total + item["size_bytes"] > max_total_bytes:
            continue
        selected.append(item)
        total += item["size_bytes"]

    if not selected:
        return {
            "status": "skipped",
            "reason": "no_tensor_fits_probe_limits",
            "max_tensor_bytes": max_tensor_bytes,
            "max_total_bytes": max_total_bytes,
            "max_tensors": max_tensors,
        }
    if not host_runtime.is_file():
        return {
            "status": "skipped",
            "reason": "host_runtime_missing",
            "host_runtime": repo_relative(host_runtime),
        }

    runtime = load_cuda_runtime(host_runtime)
    ctx = runtime.create_device_context()
    if not ctx:
        return {"status": "fail", "reason": "create_device_context_failed"}

    copied = []
    device_ptrs = []
    try:
        init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
        if init_status != 0:
            return {
                "status": "fail",
                "reason": "simpler_init_failed",
                "return_code": init_status,
            }
        for item in selected:
            data = read_tensor_bytes(item)
            buffer = ctypes.create_string_buffer(data, len(data))
            dev_ptr = runtime.device_malloc_ctx(ctx, len(data))
            if not dev_ptr:
                return {
                    "status": "fail",
                    "reason": "device_malloc_failed",
                    "tensor": item["tensor"],
                    "size_bytes": len(data),
                }
            device_ptrs.append(dev_ptr)
            copy_status = runtime.copy_to_device_ctx(
                ctx,
                dev_ptr,
                ctypes.cast(buffer, ctypes.c_void_p),
                len(data),
            )
            if copy_status != 0:
                return {
                    "status": "fail",
                    "reason": "copy_to_device_failed",
                    "tensor": item["tensor"],
                    "return_code": copy_status,
                }
            copied.append(
                {
                    "slot_id": item["slot_id"],
                    "tensor": item["tensor"],
                    "size_bytes": len(data),
                    "binding_group": item["binding_group"],
                }
            )
    except Exception as exc:  # noqa: BLE001 - artifact records the runtime error.
        return {
            "status": "fail",
            "reason": "exception",
            "message": str(exc),
        }
    finally:
        for dev_ptr in device_ptrs:
            runtime.device_free_ctx(ctx, dev_ptr)
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)

    return {
        "mode": "bounded_copy",
        "status": "pass",
        "device": device,
        "host_runtime": repo_relative(host_runtime),
        "copied_tensor_count": len(copied),
        "copied_bytes": sum(item["size_bytes"] for item in copied),
        "copied_tensors": copied,
    }


