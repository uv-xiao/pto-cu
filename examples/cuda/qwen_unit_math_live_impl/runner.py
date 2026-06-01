"""Live CUDA execution for the Qwen unit-math persistent DAG."""

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

from qwen_persistent_proxy_live_impl.runner import _copy_from_device, _copy_to_device
from qwen_persistent_proxy_live_impl.runtime import PtoRunTiming, device_name, load_runtime
from qwen_persistent_task_bodies_impl.lifecycle import task_functions
from qwen_unit_math_live_impl.graph import make_state, make_tasks
from qwen_unit_math_live_impl.plan import (
    CALLABLES,
    build_unit_math_live_plan,
    repo_relative,
)


def _selected_functions() -> list[Any]:
    wanted = {func_id for _, func_id in CALLABLES}
    return [function for function in task_functions() if function.func_id in wanted]


def _alloc(runtime: Any, ctx: Any, allocated: list[int], size: int) -> int:
    ptr = runtime.device_malloc_ctx(ctx, size)
    if not ptr:
        raise RuntimeError(f"device allocation failed for {size} bytes")
    allocated.append(int(ptr))
    return int(ptr)


def _observed(hosts: dict[str, Any]) -> dict[str, list[float]]:
    return {
        key: [round(float(value), 6) for value in hosts[key]]
        for key in ("rmsnorm", "context", "key_cache", "value_cache", "mlp", "logits")
    }


def _status(
    *,
    plan: dict[str, Any],
    observed: dict[str, list[float]],
    counters: Any,
    scheduler_processed: Any,
) -> tuple[str, float]:
    names = {
        "rmsnorm": "rmsnorm",
        "context": "attention_context",
        "key_cache": "key_cache",
        "value_cache": "value_cache",
        "mlp": "mlp_swiglu",
        "logits": "logits",
    }
    max_abs_error = max(
        abs(observed[name][index] - plan["expected"][expected_name][index])
        for name, expected_name in names.items()
        for index in range(4)
    )
    passed = (
        max_abs_error <= 1e-5
        and int(counters[4]) == 4
        and int(counters[5]) == 0
        and int(scheduler_processed[0]) >= 4
    )
    return "pass" if passed else "fail", max_abs_error


def run_unit_math_live(
    *,
    device: int = 0,
    arch: str = "compute_80",
    cache_root: Path | None = None,
    build_runtime: bool = False,
) -> dict[str, Any]:
    plan = build_unit_math_live_plan()
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

    try:
        init_status = runtime.simpler_init(ctx, device, None, 0, None, 0)
        if init_status != 0:
            raise RuntimeError(f"simpler_init failed with status {init_status}")

        array_t = ctypes.c_float * 4
        hosts = {
            "hidden": array_t(*plan["inputs"]["hidden"]),
            "norm_weight": array_t(*plan["inputs"]["norm_weight"]),
            "q_weight": array_t(*plan["inputs"]["q_proj_weight"]),
            "k_weight": array_t(*plan["inputs"]["k_proj_weight"]),
            "v_weight": array_t(*plan["inputs"]["v_proj_weight"]),
            "gate_weight": array_t(*plan["inputs"]["gate_proj_weight"]),
            "up_weight": array_t(*plan["inputs"]["up_proj_weight"]),
            "lm_head": array_t(*plan["inputs"]["lm_head_weight"]),
            "rmsnorm": array_t(*([0.0] * 4)),
            "context": array_t(*([0.0] * 4)),
            "key_cache": array_t(*([0.0] * 4)),
            "value_cache": array_t(*([0.0] * 4)),
            "mlp": array_t(*([0.0] * 4)),
            "logits": array_t(*([0.0] * 4)),
        }
        ptrs = {
            name: _alloc(runtime, ctx, allocated, ctypes.sizeof(host))
            for name, host in hosts.items()
        }
        for name in (
            "hidden",
            "norm_weight",
            "q_weight",
            "k_weight",
            "v_weight",
            "gate_weight",
            "up_weight",
            "lm_head",
        ):
            _copy_to_device(runtime, ctx, ptrs[name], hosts[name], name)

        u32_3 = ctypes.c_uint32 * 3
        u32_4 = ctypes.c_uint32 * 4
        u32_8 = ctypes.c_uint32 * 8
        u32_11 = ctypes.c_uint32 * 11
        graph_hosts = {
            "tasks": make_tasks(ptrs),
            "dependents": u32_3(1, 2, 3),
            "fanin": u32_4(0, 1, 1, 1),
            "ready_flags": u32_8(*([0] * 8)),
            "completion_flags": u32_8(*([0] * 8)),
            "counters": u32_11(*([0] * 11)),
            "scheduler_processed": (ctypes.c_uint32 * 1)(0),
        }
        for name, host in graph_hosts.items():
            ptrs[name] = _alloc(runtime, ctx, allocated, ctypes.sizeof(host))
            _copy_to_device(runtime, ctx, ptrs[name], host, name)
        ptrs["ready_queue"] = _alloc(runtime, ctx, allocated, ctypes.sizeof(graph_hosts["ready_flags"]))
        ptrs["completion_queue"] = _alloc(runtime, ctx, allocated, ctypes.sizeof(graph_hosts["completion_flags"]))
        ptrs["state"] = _alloc(runtime, ctx, allocated, ctypes.sizeof(CudaPersistentDagState))
        _copy_to_device(runtime, ctx, ptrs["state"], make_state(plan, ptrs), "state")

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

        for name in ("rmsnorm", "context", "key_cache", "value_cache", "mlp", "logits"):
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

    observed = _observed(hosts)
    status, max_abs_error = _status(
        plan=plan,
        observed=observed,
        counters=graph_hosts["counters"],
        scheduler_processed=graph_hosts["scheduler_processed"],
    )
    return {
        **plan,
        "kind": "pto_qwen_unit_math_live_execution",
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
            "completed_count": int(graph_hosts["counters"][4]),
            "error_count": int(graph_hosts["counters"][5]),
            "error_code": int(graph_hosts["counters"][6]),
            "error_task_id": int(graph_hosts["counters"][7]),
            "scheduler_processed_count": int(graph_hosts["counters"][10]),
            "scheduler_processed_by_block": [int(graph_hosts["scheduler_processed"][0])],
        },
        "timing_ns": {
            "host_wall": int(timing.host_wall_ns),
            "device_wall": int(timing.device_wall_ns),
        },
        "implemented_contracts": [
            *plan["implemented_contracts"],
            "qwen_unit_math_cuda_live_execution",
        ],
    }
