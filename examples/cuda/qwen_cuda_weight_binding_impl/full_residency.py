"""Full CUDA residency probe for Qwen weights."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from .common import repo_relative
from .cuda_runtime import (
    copy_file_range_to_device,
    load_cuda_runtime,
    verify_device_prefix,
)


def run_cuda_full_residency_probe(
    *,
    bindings: list[dict[str, Any]],
    device: int,
    host_runtime: Path,
    chunk_bytes: int,
    verify_tensors: int,
    verify_bytes: int,
) -> dict[str, Any]:
    if not host_runtime.is_file():
        return {
            "mode": "full_residency",
            "status": "skipped",
            "reason": "host_runtime_missing",
            "host_runtime": repo_relative(host_runtime),
        }
    if not bindings:
        return {
            "mode": "full_residency",
            "status": "skipped",
            "reason": "no_bindings",
        }

    runtime = load_cuda_runtime(host_runtime)
    ctx = runtime.create_device_context()
    if not ctx:
        return {
            "mode": "full_residency",
            "status": "fail",
            "reason": "create_device_context_failed",
        }

    resident = []
    device_ptrs: list[int] = []
    try:
        init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
        if init_status != 0:
            return {
                "mode": "full_residency",
                "status": "fail",
                "reason": "simpler_init_failed",
                "return_code": init_status,
            }

        for item in bindings:
            size_bytes = int(item["size_bytes"])
            dev_ptr = runtime.device_malloc_ctx(ctx, size_bytes)
            if not dev_ptr:
                return {
                    "mode": "full_residency",
                    "status": "fail",
                    "reason": "device_malloc_failed",
                    "tensor": item["tensor"],
                    "size_bytes": size_bytes,
                    "resident_tensor_count": len(resident),
                    "resident_bytes": sum(
                        record["size_bytes"] for record in resident
                    ),
                }
            ptr_value = int(dev_ptr)
            device_ptrs.append(ptr_value)
            copy_file_range_to_device(
                runtime=runtime,
                ctx=ctx,
                binding=item,
                dev_ptr=ptr_value,
                chunk_bytes=chunk_bytes,
            )
            resident.append(
                {
                    "slot_id": item["slot_id"],
                    "tensor": item["tensor"],
                    "size_bytes": size_bytes,
                    "binding_group": item["binding_group"],
                }
            )

        verified = []
        small_first = sorted(resident, key=lambda item: (item["size_bytes"], item["tensor"]))
        resident_by_slot = {item["slot_id"]: item for item in bindings}
        ptr_by_slot = {
            item["slot_id"]: ptr for item, ptr in zip(bindings, device_ptrs, strict=True)
        }
        for record in small_first[:verify_tensors]:
            binding = resident_by_slot[record["slot_id"]]
            if not verify_device_prefix(
                runtime=runtime,
                ctx=ctx,
                binding=binding,
                dev_ptr=ptr_by_slot[record["slot_id"]],
                verify_bytes=verify_bytes,
            ):
                return {
                    "mode": "full_residency",
                    "status": "fail",
                    "reason": "verification_mismatch",
                    "tensor": record["tensor"],
                }
            verified.append(
                {
                    "slot_id": record["slot_id"],
                    "tensor": record["tensor"],
                    "verified_bytes": min(verify_bytes, record["size_bytes"]),
                    "binding_group": record["binding_group"],
                }
            )
    except Exception as exc:  # noqa: BLE001 - artifact records the runtime error.
        return {
            "mode": "full_residency",
            "status": "fail",
            "reason": "exception",
            "message": str(exc),
            "resident_tensor_count": len(resident),
            "resident_bytes": sum(record["size_bytes"] for record in resident),
        }
    finally:
        for dev_ptr in reversed(device_ptrs):
            runtime.device_free_ctx(ctx, ctypes.c_void_p(dev_ptr))
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)

    return {
        "mode": "full_residency",
        "status": "pass",
        "device": device,
        "host_runtime": repo_relative(host_runtime),
        "copy_chunk_bytes": chunk_bytes,
        "resident_tensor_count": len(resident),
        "resident_bytes": sum(item["size_bytes"] for item in resident),
        "freed_tensor_count": len(device_ptrs),
        "verified_tensor_count": len(verified),
        "verified_tensors": verified,
        "resident_bindings_sample": resident[:16],
    }

