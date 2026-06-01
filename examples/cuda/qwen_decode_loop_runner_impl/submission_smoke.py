"""CUDA-live smoke for the Qwen submission task-function set."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagArgs,
    CudaPersistentDagState,
    compile_cuda_persistent_device,
    prepare_cuda_persistent_device_callable,
)

from qwen_decode_loop_runner_impl.submission import QWEN_TASK_FUNCTIONS
from qwen_decode_loop_runner_impl.submission_smoke_graph import (
    expected_outputs,
    make_state,
    make_tasks,
    smoke_inputs,
)
from qwen_persistent_proxy_live_impl.runner import _copy_from_device, _copy_to_device
from qwen_persistent_proxy_live_impl.runtime import (
    PtoRunTiming,
    device_name,
    load_runtime,
)
from qwen_persistent_task_bodies_impl.lifecycle import task_functions


ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _alloc(runtime: Any, ctx: Any, allocated: list[int], size: int) -> int:
    ptr = runtime.device_malloc_ctx(ctx, size)
    if not ptr:
        raise RuntimeError(f"device allocation failed for {size} bytes")
    allocated.append(int(ptr))
    return int(ptr)


def _selected_functions() -> list[Any]:
    wanted = {item["func_id"] for item in QWEN_TASK_FUNCTIONS}
    return [function for function in task_functions() if function.func_id in wanted]


def run_submission_smoke(
    *,
    device: int = 0,
    arch: str = "compute_80",
    cache_root: Path | None = None,
    build_runtime: bool = False,
) -> dict[str, Any]:
    plan = {
        "scheduler_blocks": 1,
        "worker_blocks": 1,
        "block_dim": 64,
        "queue_capacity": 16,
    }
    runtime, binaries = load_runtime(build_runtime=build_runtime)
    artifact = compile_cuda_persistent_device(
        _selected_functions(),
        arch=arch,
        cache_root=cache_root,
    )
    ctx = runtime.create_device_context()
    if not ctx:
        raise RuntimeError("create_device_context returned null")
    allocated: list[int] = []
    callable_prepared = False
    inputs = smoke_inputs()

    try:
        init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
        if init_status != 0:
            raise RuntimeError(f"simpler_init failed with status {init_status}")

        float4 = ctypes.c_float * 4
        uint4 = ctypes.c_uint32 * 4
        hosts: dict[str, Any] = {"token_ids": uint4(*inputs["token_ids"])}
        for name, values in inputs.items():
            if name != "token_ids":
                hosts[name] = float4(*values)
        for name in [f"x{index}" for index in range(9)] + [
            "logits",
            "key_cache",
            "value_cache",
        ]:
            hosts[name] = float4(*([0.0] * 4))
        ptrs = {
            name: _alloc(runtime, ctx, allocated, ctypes.sizeof(host))
            for name, host in hosts.items()
        }
        for name, host in hosts.items():
            if name in inputs:
                _copy_to_device(runtime, ctx, ptrs[name], host, name)

        u32_10 = ctypes.c_uint32 * 10
        u32_11 = ctypes.c_uint32 * 11
        u32_16 = ctypes.c_uint32 * 16
        graph_hosts = {
            "tasks": make_tasks(ptrs),
            "dependents": u32_10(1, 2, 3, 4, 5, 6, 7, 8, 9, 0),
            "fanin": u32_10(0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            "ready_flags": u32_16(*([0] * 16)),
            "completion_flags": u32_16(*([0] * 16)),
            "counters": u32_11(*([0] * 11)),
            "scheduler_processed": (ctypes.c_uint32 * 1)(0),
        }
        for name, host in graph_hosts.items():
            ptrs[name] = _alloc(runtime, ctx, allocated, ctypes.sizeof(host))
            _copy_to_device(runtime, ctx, ptrs[name], host, name)
        ptrs["ready_queue"] = _alloc(
            runtime,
            ctx,
            allocated,
            ctypes.sizeof(graph_hosts["ready_flags"]),
        )
        ptrs["completion_queue"] = _alloc(
            runtime,
            ctx,
            allocated,
            ctypes.sizeof(graph_hosts["completion_flags"]),
        )
        ptrs["state"] = _alloc(
            runtime,
            ctx,
            allocated,
            ctypes.sizeof(CudaPersistentDagState),
        )
        _copy_to_device(runtime, ctx, ptrs["state"], make_state(plan, ptrs), "state")

        prepared = prepare_cuda_persistent_device_callable(
            artifact,
            grid_dim=plan["scheduler_blocks"] + plan["worker_blocks"],
            block_dim=plan["block_dim"],
        )
        if runtime.prepare_callable(ctx, 0, prepared.byref()) != 0:
            raise RuntimeError("prepare_callable failed")
        callable_prepared = True

        timing = PtoRunTiming()
        args = CudaPersistentDagArgs(state=ptrs["state"])
        status = runtime.run_prepared(
            ctx,
            None,
            0,
            ctypes.byref(args),
            plan["block_dim"],
            0,
            0,
            0,
            0,
            0,
            None,
            ctypes.byref(timing),
        )
        if status != 0:
            raise RuntimeError(f"run_prepared failed with status {status}")

        for name in ("logits", "key_cache", "value_cache"):
            _copy_from_device(runtime, ctx, hosts[name], ptrs[name], name)
        for name in ("counters", "scheduler_processed"):
            _copy_from_device(runtime, ctx, graph_hosts[name], ptrs[name], name)
    finally:
        if callable_prepared:
            runtime.unregister_callable(ctx, 0)
        for ptr in reversed(allocated):
            runtime.device_free_ctx(ctx, ctypes.c_void_p(ptr))
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)

    observed = {
        name: [round(float(value), 6) for value in hosts[name]]
        for name in ("logits", "key_cache", "value_cache")
    }
    expected = expected_outputs(inputs)
    max_abs_error = max(
        abs(observed[name][index] - expected[name][index])
        for name in expected
        for index in range(4)
    )
    counters = graph_hosts["counters"]
    scheduler_processed = graph_hosts["scheduler_processed"]
    status = (
        "pass"
        if max_abs_error <= 1e-5
        and int(counters[4]) == 10
        and int(counters[5]) == 0
        and int(scheduler_processed[0]) >= 10
        else "fail"
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_submission_smoke_execution",
        "status": status,
        "runtime": "cuda/persistent_device",
        "serving_coverage": "diagnostic_qwen_descriptor_smoke",
        "device": {"ordinal": device, "name": device_name(device), "arch": arch},
        "runtime_binary": repo_relative(Path(binaries.host_path)),
        "artifact": {
            "cache_key": artifact.cache_key,
            "cache_hit": artifact.cache_hit,
            "source_path": repo_relative(artifact.source_path),
            "ptx_path": repo_relative(artifact.ptx_path),
            "entry_name": artifact.entry_name,
            "source_kind": artifact.source_kind,
        },
        "func_id_sequence": [item["func_id"] for item in QWEN_TASK_FUNCTIONS],
        "observed": observed,
        "expected": expected,
        "max_abs_error": max_abs_error,
        "scheduler_counters": {
            "completed_count": int(counters[4]),
            "error_count": int(counters[5]),
            "error_code": int(counters[6]),
            "error_task_id": int(counters[7]),
            "scheduler_processed_count": int(counters[10]),
            "scheduler_processed_by_block": [int(scheduler_processed[0])],
        },
        "timing_ns": {
            "host_wall": int(timing.host_wall_ns),
            "device_wall": int(timing.device_wall_ns),
        },
        "implemented_contracts": ["qwen_decode_loop_submission_smoke_execution"],
        "remaining_runtime_gaps": [
            "resource_backed_full_qwen_decode_loop_execution",
            "full_qwen_numerical_correctness",
            "viewer_result_import",
        ],
    }
