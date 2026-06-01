"""Materialize persistent decode arguments within token pointer lifetime."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from qwen_token_pointer_table_impl.common import (
    DECODE_ARGS_SCRIPT,
    DEFAULT_HOST_RUNTIME,
    load_module,
    write_json,
)
from qwen_token_pointer_table_impl.pointers import (
    build_token_binding,
    dry_run_pointer_table,
    live_pointer_table,
)


def materialize_decode_args(
    *,
    token_binding: dict[str, Any],
    pointer_table: dict[str, Any],
) -> dict[str, Any]:
    decode_module = load_module(DECODE_ARGS_SCRIPT, "qwen_decode_args_for_token_ptrs")
    with tempfile.TemporaryDirectory(prefix="pto-qwen-token-pointers-") as tmp:
        tmpdir = Path(tmp)
        token_binding_json = tmpdir / "qwen-cuda-token-buffer-binding.json"
        pointer_table_json = tmpdir / "qwen-token-pointer-table.json"
        write_json(token_binding_json, token_binding)
        write_json(pointer_table_json, pointer_table)
        manifest = decode_module.build_decode_arg_manifest(
            cuda_token_buffer_json=token_binding_json,
            token_pointer_table_json=pointer_table_json,
        )
    manifest["cuda_token_buffer_json"] = (
        "generated_from_examples/cuda/qwen_cuda_token_buffer_binding.py"
    )
    manifest["token_pointer_table_json"] = (
        "owned_by_examples/cuda/qwen_token_pointer_table.py"
    )
    return manifest


def build_token_pointer_table_lifecycle(
    *,
    mode: str = "offline",
    cache_dir: Path | None = None,
    cuda_live: bool = False,
    device: int = 0,
    host_runtime: Path = DEFAULT_HOST_RUNTIME,
    pointer_base: int = 0x30000000,
    pointer_stride: int = 0x10000,
) -> dict[str, Any]:
    token_binding = build_token_binding(mode=mode, cache_dir=cache_dir)
    if cuda_live:
        pointer_table, close = live_pointer_table(
            token_binding=token_binding,
            mode=mode,
            cache_dir=cache_dir,
            host_runtime=host_runtime,
            device=device,
        )
    else:
        pointer_table = dry_run_pointer_table(
            token_binding=token_binding,
            pointer_base=pointer_base,
            pointer_stride=pointer_stride,
        )
        close = lambda: pointer_table.get("pointer_count", 0)

    if pointer_table.get("status") == "cuda_token_pointer_table_ready":
        try:
            decode_args = materialize_decode_args(
                token_binding=token_binding,
                pointer_table=pointer_table,
            )
        finally:
            freed = close()
    else:
        decode_args = {}
        freed = close()
    closed = {
        **pointer_table,
        "status": "cuda_token_pointer_table_closed",
        "freed_pointer_count": freed,
    }
    return lifecycle_payload(
        cuda_live=cuda_live,
        token_binding=token_binding,
        pointer_table=pointer_table,
        closed_pointer_table=closed,
        decode_args=decode_args,
    )


def lifecycle_payload(
    *,
    cuda_live: bool,
    token_binding: dict[str, Any],
    pointer_table: dict[str, Any],
    closed_pointer_table: dict[str, Any],
    decode_args: dict[str, Any],
) -> dict[str, Any]:
    ready = (
        pointer_table.get("status") == "cuda_token_pointer_table_ready"
        and decode_args.get("status") == "persistent_decode_args_ready"
        and closed_pointer_table.get("freed_pointer_count")
        == pointer_table.get("pointer_count")
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_cuda_token_pointer_table_lifecycle",
        "status": (
            "token_pointer_table_lifecycle_ready"
            if ready
            else "token_pointer_table_lifecycle_incomplete"
        ),
        "mode": "cuda_live" if cuda_live else "dry_run_pointer_lifecycle",
        "cuda_token_buffer_status": token_binding.get("status"),
        "pointer_table": pointer_table,
        "closed_pointer_table": closed_pointer_table,
        "decode_args": decode_args,
        "pointer_count": pointer_table.get("pointer_count", 0),
        "freed_pointer_count": closed_pointer_table.get("freed_pointer_count", 0),
        "implemented_contracts": [
            "live_token_pointer_table_owner",
            "persistent_decode_arg_materialization_during_lifetime",
            "cuda_live_token_pointer_table" if cuda_live else "dry_run_pointer_lifecycle",
        ],
        "remaining_runtime_gaps": [
            "qwen_kernel_token_consumption",
            "decode_loop_execution",
        ],
    }
