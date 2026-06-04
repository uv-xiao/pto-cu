"""Single CUDA-context resource session for Qwen launch preflight."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import qwen_cuda_weight_binding as weight_runtime
from qwen_decode_loop_runner_impl.activation_workspace import DEFAULT_HOST_RUNTIME
from qwen_decode_loop_runner_impl.single_context_allocators import (
    open_kv_table,
    open_resident_table,
    open_token_table,
    open_workspace,
)
from qwen_kv_cache_binding_impl.lifecycle import build_kv_cache_lifecycle
from qwen_resident_weight_table_impl.lifecycle import lifecycle_payload
from qwen_token_pointer_table_impl.lifecycle import (
    lifecycle_payload as token_lifecycle_payload,
)


class SingleContextLiveSession:
    def __init__(
        self,
        *,
        mode: str,
        cache_dir: Path | None,
        device: int,
        host_runtime: Path | None,
        workload_ids: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.cache_dir = cache_dir
        self.device = device
        self.host_runtime = host_runtime or DEFAULT_HOST_RUNTIME
        self.workload_ids = workload_ids
        self.runtime: Any | None = None
        self.ctx: Any | None = None
        self.allocations: list[tuple[str, int]] = []
        self.freed_by_group: dict[str, int] = {}
        self.token_binding: dict[str, Any] = {}
        self.token_table: dict[str, Any] = {}
        self.decode_args: dict[str, Any] = {}
        self.kv_bindings: list[dict[str, Any]] = []
        self.kv_table: dict[str, Any] = {}
        self.resident_table: dict[str, Any] = {}
        self.resident_materialization: dict[str, Any] = {}
        self.binding_source = ""
        self.weight_args_path: Path | None = None
        self.activation_workspace: dict[str, Any] = {}

    def open(self) -> None:
        self.runtime = weight_runtime.load_cuda_runtime(self.host_runtime)
        self.ctx = self.runtime.create_device_context()
        if not self.ctx:
            raise RuntimeError("create_device_context failed")
        status = self.runtime.simpler_init(self.ctx, self.device, None, 0, None, 0)
        if status != 0:
            self.runtime.destroy_device_context(self.ctx)
            raise RuntimeError(f"simpler_init failed with status {status}")
        try:
            self.open_tables()
        except Exception:
            self.close()
            raise

    def open_tables(self) -> None:
        assert self.runtime is not None and self.ctx is not None
        self.token_binding, self.token_table, self.decode_args = open_token_table(
            runtime=self.runtime,
            ctx=self.ctx,
            mode=self.mode,
            cache_dir=self.cache_dir,
            host_runtime=self.host_runtime,
            device=self.device,
            allocations=self.allocations,
            workload_ids=self.workload_ids,
        )
        self.kv_bindings, self.kv_table = open_kv_table(
            runtime=self.runtime,
            ctx=self.ctx,
            host_runtime=self.host_runtime,
            device=self.device,
            allocations=self.allocations,
            workload_ids=self.workload_ids,
        )
        (
            self.binding_source,
            self.weight_args_path,
            self.resident_table,
            self.resident_materialization,
        ) = open_resident_table(
            runtime=self.runtime,
            ctx=self.ctx,
            host_runtime=self.host_runtime,
            device=self.device,
            allocations=self.allocations,
        )

    def open_activation_workspace(
        self,
        *,
        plans: list[dict[str, Any]],
        graph_task_count: int,
        descriptors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        assert self.runtime is not None and self.ctx is not None
        self.activation_workspace = open_workspace(
            runtime=self.runtime,
            ctx=self.ctx,
            host_runtime=self.host_runtime,
            device=self.device,
            plans=plans,
            graph_task_count=graph_task_count,
            descriptors=descriptors,
            allocations=self.allocations,
        )
        return self.activation_workspace

    def token_lifecycle(self) -> dict[str, Any]:
        return token_lifecycle_payload(
            cuda_live=True,
            token_binding=self.token_binding,
            pointer_table=self.token_table,
            closed_pointer_table=self.closed_table(
                self.token_table,
                "token_pointer_table",
            ),
            decode_args=self.decode_args,
        )

    def kv_lifecycle(self) -> dict[str, Any]:
        payload = build_kv_cache_lifecycle(cuda_live=False)
        payload.update(
            {
                "status": "kv_cache_lifecycle_ready",
                "mode": "cuda_live",
                "pointer_table": self.kv_table,
                "closed_pointer_table": self.closed_table(self.kv_table, "kv_cache"),
                "kv_cache_bindings": self.kv_bindings,
                "pointer_count": self.kv_table.get("pointer_count", 0),
                "total_byte_count": self.kv_table.get("total_byte_count", 0),
                "implemented_contracts": [
                    "kv_cache_key_value_field_binding",
                    "kv_cache_token_position_lifecycle",
                    "cuda_live_kv_cache_owner",
                    "single_context_session_kv_cache",
                ],
            }
        )
        return payload

    def resident_lifecycle(self) -> dict[str, Any]:
        return lifecycle_payload(
            dry_run=False,
            binding_source=self.binding_source,
            weight_args_path=self.weight_args_path,
            pointer_table=self.resident_table,
            closed_pointer_table=self.closed_table(
                self.resident_table,
                "resident_weight_table",
            ),
            materialization=self.resident_materialization,
        )

    def close(self) -> dict[str, Any]:
        if self.runtime is None or self.ctx is None:
            return {}
        for group, ptr in reversed(self.allocations):
            self.runtime.device_free_ctx(self.ctx, ctypes.c_void_p(ptr))
            self.freed_by_group[group] = self.freed_by_group.get(group, 0) + 1
        self.runtime.finalize_device(self.ctx)
        self.runtime.destroy_device_context(self.ctx)
        self.runtime = None
        self.ctx = None
        return {
            "status": "single_context_session_closed",
            "freed_pointer_count": sum(self.freed_by_group.values()),
            "freed_by_group": dict(self.freed_by_group),
        }

    def closed_table(self, table: dict[str, Any], group: str) -> dict[str, Any]:
        return {
            **table,
            "status": table["status"].replace("_ready", "_closed"),
            "freed_pointer_count": self.freed_by_group.get(group, 0),
        }


def open_single_context_live_session(
    *,
    mode: str,
    cache_dir: Path | None,
    device: int,
    host_runtime: Path | None,
    workload_ids: list[str] | None = None,
) -> SingleContextLiveSession:
    session = SingleContextLiveSession(
        mode=mode,
        cache_dir=cache_dir,
        device=device,
        host_runtime=host_runtime,
        workload_ids=workload_ids,
    )
    session.open()
    return session
