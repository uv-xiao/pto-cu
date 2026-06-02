"""Live CUDA persistent-device execution for the Qwen QKV proxy."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagArgs,
    CudaPersistentDagState,
    CudaPersistentDagTask,
    compile_cuda_persistent_device,
    prepare_cuda_persistent_device_callable,
)

from qwen_persistent_proxy_live_impl.plan import (
    FUNC_ID,
    build_live_proxy_plan,
    repo_relative,
)
from qwen_persistent_proxy_live_impl.runtime import (
    PtoRunTiming,
    device_name,
    load_runtime,
)
from qwen_persistent_task_bodies_impl.lifecycle import task_functions


def _copy_to_device(runtime: Any, ctx: Any, ptr: int, host: Any, label: str) -> None:
    status = runtime.copy_to_device_ctx(ctx, ptr, ctypes.byref(host), ctypes.sizeof(host))
    if status != 0:
        raise RuntimeError(f"copy_to_device {label} failed")


def _copy_from_device(runtime: Any, ctx: Any, host: Any, ptr: int, label: str) -> None:
    status = runtime.copy_from_device_ctx(ctx, ctypes.byref(host), ptr, ctypes.sizeof(host))
    if status != 0:
        raise RuntimeError(f"copy_from_device {label} failed")


def _single_qkv_function() -> Any:
    return next(function for function in task_functions() if function.func_id == FUNC_ID)


def _make_task(ptrs: dict[str, int]) -> Any:
    tensor_args_t = ctypes.c_void_p * 5
    scalar_args_t = ctypes.c_float * 4
    task_t = CudaPersistentDagTask * 1
    return task_t(
        CudaPersistentDagTask(
            func_id=FUNC_ID,
            a=ptrs["a"],
            b=ptrs["b"],
            out=ptrs["out"],
            n=4,
            dependent_begin=0,
            dependent_count=0,
            initial_fanin=0,
            scalar0=0.0,
            scalar1=0.0,
            rows=0,
            cols=0,
            inner=0,
            lda=0,
            ldb=0,
            ldc=0,
            a_batch_stride=0,
            b_batch_stride=0,
            out_batch_stride=0,
            c=ptrs["c"],
            d=ptrs["d"],
            tensor_args=tensor_args_t(ptrs["weight"], None, None, None, None),
            scalar_args=scalar_args_t(0.0, 0.0, 0.0, 0.0),
            tensor_arg_count=1,
            scalar_arg_count=0,
        )
    )


def _make_state(plan: dict[str, Any], ptrs: dict[str, int]) -> CudaPersistentDagState:
    word = ctypes.sizeof(ctypes.c_uint32)
    return CudaPersistentDagState(
        tasks=ptrs["tasks"],
        task_count=1,
        dependents=ptrs["dependents"],
        dependent_count=0,
        fanin=ptrs["fanin"],
        ready_queue=ptrs["ready_queue"],
        ready_flags=ptrs["ready_flags"],
        completion_queue=ptrs["completion_queue"],
        completion_flags=ptrs["completion_flags"],
        queue_capacity=plan["dag"]["queue_capacity"],
        queue_head=ptrs["counters"],
        queue_tail=ptrs["counters"] + word,
        completion_head=ptrs["counters"] + 2 * word,
        completion_tail=ptrs["counters"] + 3 * word,
        completed_count=ptrs["counters"] + 4 * word,
        error_count=ptrs["counters"] + 5 * word,
        error_code=ptrs["counters"] + 6 * word,
        error_task_id=ptrs["counters"] + 7 * word,
        scheduler_blocks=plan["dag"]["scheduler_blocks"],
        scheduler_init_count=ptrs["counters"] + 8 * word,
        scheduler_loop_count=ptrs["counters"] + 9 * word,
        scheduler_processed_count=ptrs["counters"] + 10 * word,
        scheduler_processed_by_block=ptrs["scheduler_processed"],
    )


def _status_payload(
    *,
    plan: dict[str, Any],
    observed: dict[str, list[float]],
    counters: Any,
    scheduler_processed: Any,
) -> tuple[str, float]:
    expected = plan["expected"]
    max_abs_error = max(
        abs(observed[name][index] - expected[name][index])
        for name in ("out", "c", "d")
        for index in range(4)
    )
    passed = (
        max_abs_error == 0.0
        and int(counters[4]) == 1
        and int(counters[5]) == 0
        and int(scheduler_processed[0]) >= 1
    )
    return "pass" if passed else "fail", max_abs_error


def run_live_proxy(
    *,
    device: int = 0,
    arch: str = "compute_80",
    cache_root: Path | None = None,
    build_runtime: bool = False,
) -> dict[str, Any]:
    plan = build_live_proxy_plan()
    runtime, binaries = load_runtime(build_runtime=build_runtime)
    artifact = compile_cuda_persistent_device(
        [_single_qkv_function()],
        arch=arch,
        cache_root=cache_root,
    )

    ctx = runtime.create_device_context()
    if not ctx:
        raise RuntimeError("create_device_context returned null")
    allocated: list[int] = []
    callable_prepared = False

    try:
        init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
        if init_status != 0:
            raise RuntimeError(f"simpler_init failed with status {init_status}")

        array_t = ctypes.c_float * 4
        hosts = {
            "a": array_t(*plan["inputs"]["a"]),
            "b": array_t(*plan["inputs"]["b"]),
            "c": array_t(*plan["inputs"]["c"]),
            "d": array_t(*plan["inputs"]["d"]),
            "out": array_t(*([0.0] * 4)),
            "weight": array_t(*plan["inputs"]["weights"][0]),
        }

        def alloc(size: int) -> int:
            ptr = runtime.device_malloc_ctx(ctx, size)
            if not ptr:
                raise RuntimeError(f"device allocation failed for {size} bytes")
            allocated.append(int(ptr))
            return int(ptr)

        ptrs = {
            name: alloc(ctypes.sizeof(host))
            for name, host in hosts.items()
        }
        for name in ("a", "b", "c", "d", "weight"):
            _copy_to_device(runtime, ctx, ptrs[name], hosts[name], name)

        u32_1 = ctypes.c_uint32 * 1
        u32_4 = ctypes.c_uint32 * 4
        u32_11 = ctypes.c_uint32 * 11
        graph_hosts = {
            "tasks": _make_task(ptrs),
            "dependents": u32_1(0),
            "fanin": u32_1(0),
            "ready_flags": u32_4(0, 0, 0, 0),
            "completion_flags": u32_4(0, 0, 0, 0),
            "counters": u32_11(*([0] * 11)),
            "scheduler_processed": u32_1(0),
        }
        for name, host in graph_hosts.items():
            ptrs[name] = alloc(ctypes.sizeof(host))
            _copy_to_device(runtime, ctx, ptrs[name], host, name)
        ptrs["ready_queue"] = alloc(ctypes.sizeof(graph_hosts["ready_flags"]))
        ptrs["completion_queue"] = alloc(ctypes.sizeof(graph_hosts["completion_flags"]))
        ptrs["state"] = alloc(ctypes.sizeof(CudaPersistentDagState))
        state = _make_state(plan, ptrs)
        _copy_to_device(runtime, ctx, ptrs["state"], state, "state")

        prepared = prepare_cuda_persistent_device_callable(
            artifact,
            grid_dim=plan["dag"]["scheduler_blocks"] + plan["dag"]["worker_blocks"],
            block_dim=plan["dag"]["block_dim"],
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
            plan["dag"]["block_dim"],
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

        for name in ("out", "c", "d"):
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
        for name in ("out", "c", "d")
    }
    counters = graph_hosts["counters"]
    scheduler_processed = graph_hosts["scheduler_processed"]
    status, max_abs_error = _status_payload(
        plan=plan,
        observed=observed,
        counters=counters,
        scheduler_processed=scheduler_processed,
    )
    return {
        **plan,
        "kind": "pto_qwen_proxy_live_execution",
        "status": status,
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
        "observed": observed,
        "max_abs_error": max_abs_error,
        "scheduler_counters": {
            "completed_count": int(counters[4]),
            "error_count": int(counters[5]),
            "error_code": int(counters[6]),
            "error_task_id": int(counters[7]),
            "scheduler_init_count": int(counters[8]),
            "scheduler_loop_count": int(counters[9]),
            "scheduler_processed_count": int(counters[10]),
            "scheduler_processed_by_block": [int(scheduler_processed[0])],
        },
        "timing_ns": {
            "host_wall": int(timing.host_wall_ns),
            "device_wall": int(timing.device_wall_ns),
        },
        "implemented_contracts": [
            *plan["implemented_contracts"],
            "controlled_proxy_live_cuda_execution",
        ],
    }
