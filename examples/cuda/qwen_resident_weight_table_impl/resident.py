"""Resident pointer owner for Qwen weights."""

from __future__ import annotations

from typing import Any, Callable

from qwen_resident_weight_table_impl.common import MODEL_ID, MODEL_REVISION


class ResidentWeightTableOwner:
    """Owns resident weight pointers for one process-scoped decode runner."""

    def __init__(
        self,
        *,
        bindings: list[dict[str, Any]],
        allocate_and_copy: Callable[[dict[str, Any]], int],
        free_pointer: Callable[[int, dict[str, Any]], None],
        device: int | str,
        source: str,
    ) -> None:
        self._bindings = list(bindings)
        self._allocate_and_copy = allocate_and_copy
        self._free_pointer = free_pointer
        self._device = device
        self._source = source
        self._pointers: list[dict[str, Any]] = []
        self._freed: list[dict[str, Any]] = []
        self._closed = False
        self._opened = False

    def __enter__(self) -> "ResidentWeightTableOwner":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def open(self) -> None:
        if self._opened and not self._closed:
            return
        if self._closed:
            raise RuntimeError("resident weight table owner is already closed")
        try:
            for item in self._bindings:
                ptr = int(self._allocate_and_copy(item))
                if ptr <= 0:
                    raise RuntimeError(
                        f"allocated pointer for {item['tensor']} is invalid"
                    )
                self._pointers.append(pointer_record(item, ptr))
        except Exception:
            self.close()
            raise
        self._opened = True

    def close(self) -> None:
        if self._closed:
            return
        for pointer in reversed(self._pointers):
            self._free_pointer(int(pointer["device_ptr"]), pointer)
            self._freed.append(pointer)
        self._pointers = []
        self._closed = True

    def pointer_table(self) -> dict[str, Any]:
        status = pointer_table_status(opened=self._opened, closed=self._closed)
        pointers = [] if self._closed else list(self._pointers)
        return {
            "schema_version": 1,
            "kind": "pto_qwen_resident_weight_pointer_table",
            "status": status,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": self._device,
            "source": self._source,
            "lifetime": pointer_table_lifetime(
                opened=self._opened,
                closed=self._closed,
            ),
            "pointer_count": len(pointers),
            "freed_pointer_count": len(self._freed),
            "resident_bytes": sum(int(item["size_bytes"]) for item in pointers),
            "pointers": pointers,
        }


def pointer_record(item: dict[str, Any], ptr: int) -> dict[str, Any]:
    return {
        "slot_id": item["slot_id"],
        "tensor": item["tensor"],
        "device_ptr": ptr,
        "device_ptr_hex": f"0x{ptr:x}",
        "size_bytes": int(item["size_bytes"]),
        "binding_group": item.get("binding_group", ""),
    }


def pointer_table_status(*, opened: bool, closed: bool) -> str:
    if closed:
        return "resident_weight_pointer_table_closed"
    if opened:
        return "resident_weight_pointer_table_ready"
    return "resident_weight_pointer_table_not_open"


def pointer_table_lifetime(*, opened: bool, closed: bool) -> str:
    if closed:
        return "closed"
    if opened:
        return "valid_until_owner_close"
    return "not_open"


def dry_run_owner(
    *,
    bindings: list[dict[str, Any]],
    pointer_base: int,
    pointer_stride: int,
) -> ResidentWeightTableOwner:
    def allocate_and_copy(item: dict[str, Any]) -> int:
        return pointer_base + int(item["slot_id"]) * pointer_stride

    def free_pointer(_ptr: int, _item: dict[str, Any]) -> None:
        return None

    return ResidentWeightTableOwner(
        bindings=bindings,
        allocate_and_copy=allocate_and_copy,
        free_pointer=free_pointer,
        device="dry_run",
        source="dry_run_pointer_lifecycle",
    )
