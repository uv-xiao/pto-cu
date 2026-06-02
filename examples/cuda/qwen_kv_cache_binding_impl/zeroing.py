"""CUDA KV-cache initialization helpers."""

from __future__ import annotations

import ctypes
from typing import Any


KV_ZERO_CHUNK_BYTES = 64 * 1024 * 1024


def zero_device_allocation(
    *,
    runtime: Any,
    ctx: Any,
    ptr: int,
    byte_count: int,
    chunk_bytes: int = KV_ZERO_CHUNK_BYTES,
) -> None:
    zero_chunk = ctypes.create_string_buffer(chunk_bytes)
    offset = 0
    while offset < byte_count:
        chunk = min(chunk_bytes, byte_count - offset)
        status = runtime.copy_to_device_ctx(
            ctx,
            ptr + offset,
            ctypes.cast(zero_chunk, ctypes.c_void_p),
            chunk,
        )
        if status != 0:
            raise RuntimeError("copy_to_device failed while zeroing KV cache")
        offset += chunk
