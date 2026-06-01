"""CUDA-backed resident weight table owner."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from qwen_resident_weight_table_impl.common import ROOT, load_module, repo_relative
from qwen_resident_weight_table_impl.resident import ResidentWeightTableOwner


class CudaResidentWeightTableOwner(ResidentWeightTableOwner):
    """Owns real CUDA allocations behind the resident pointer table."""

    def __init__(
        self,
        *,
        bindings: list[dict[str, Any]],
        device: int,
        host_runtime: Path,
        copy_chunk_bytes: int,
    ) -> None:
        module = load_module(
            ROOT / "examples" / "cuda" / "qwen_cuda_weight_binding.py",
            "qwen_cuda_weight_binding_runtime",
        )
        self._runtime = module.load_cuda_runtime(host_runtime)
        self._ctx = self._runtime.create_device_context()
        if not self._ctx:
            raise RuntimeError("failed to create CUDA device context")
        status = self._runtime.simpler_init(self._ctx, device, None, 0, None, 0)
        if status != 0:
            self._runtime.destroy_device_context(self._ctx)
            raise RuntimeError(f"simpler_init failed with status {status}")

        def allocate_and_copy(item: dict[str, Any]) -> int:
            ptr = self._runtime.device_malloc_ctx(
                self._ctx,
                int(item["size_bytes"]),
            )
            if not ptr:
                raise RuntimeError(f"device allocation failed for {item['tensor']}")
            ptr_value = int(ptr)
            module.copy_file_range_to_device(
                runtime=self._runtime,
                ctx=self._ctx,
                binding=item,
                dev_ptr=ptr_value,
                chunk_bytes=copy_chunk_bytes,
            )
            return ptr_value

        def free_pointer(ptr: int, _item: dict[str, Any]) -> None:
            self._runtime.device_free_ctx(self._ctx, ctypes.c_void_p(ptr))

        super().__init__(
            bindings=bindings,
            allocate_and_copy=allocate_and_copy,
            free_pointer=free_pointer,
            device=device,
            source=repo_relative(host_runtime),
        )

    def close(self) -> None:
        already_closed = self._closed
        super().close()
        if not already_closed:
            self._runtime.finalize_device(self._ctx)
            self._runtime.destroy_device_context(self._ctx)
