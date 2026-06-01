"""Live CUDA persistent-device execution for the Qwen microdecode proxy."""

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

from qwen_persistent_microdecode_live_impl.graph import make_state, make_tasks
from qwen_persistent_microdecode_live_impl.plan import (
    CALLABLES,
    build_live_microdecode_plan,
    repo_relative,
)
from qwen_persistent_proxy_live_impl.runner import _copy_from_device, _copy_to_device
from qwen_persistent_proxy_live_impl.runtime import PtoRunTiming, device_name, load_runtime
from qwen_persistent_task_bodies_impl.lifecycle import task_functions


def _selected_functions() -> list[Any]:
    wanted = {func_id for _, func_id in CALLABLES}
    return [function for function in task_functions() if function.func_id in wanted]


def _alloc(runtime: Any, ctx: Any, allocated: list[int], size: int) -> int:
    ptr = runtime.device_malloc_ctx(ctx, size)
    if not ptr:
        raise RuntimeError(f"device allocation failed for {size} bytes")
    allocated.append(int(ptr))
    return int(ptr)


def _status(
    *,
    expected: dict[str, list[float]],
    observed: dict[str, list[float]],
    counters: Any,
    scheduler_processed: Any,
) -> tuple[str, float]:
    max_abs_error = max(
        abs(observed[name][index] - expected[name][index])
        for name in ("attention_qkv_out", "attention_o_out", "logits_out", "c", "d")
        for index in range(4)
    )
    passed = (
        max_abs_error == 0.0
        and int(counters[4]) == 3
        and int(counters[5]) == 0
        and int(scheduler_processed[0]) >= 3
    )
    return "pass" if passed else "fail", max_abs_error


def run_live_microdecode(
    *,
    device: int = 0,
    arch: str = "compute_80",
    cache_root: Path | None = None,
    build_runtime: bool = False,
    repeat_runs: int = 1,
) -> dict[str, Any]:
    plan = build_live_microdecode_plan(repeat_runs=repeat_runs)
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
            "a": array_t(*plan["inputs"]["a"]),
            "b": array_t(*plan["inputs"]["b"]),
            "c": array_t(*plan["inputs"]["c"]),
            "d": array_t(*plan["inputs"]["d"]),
            "q_weight": array_t(*plan["inputs"]["weights"][0]),
            "o_weight": array_t(*plan["inputs"]["weights"][1]),
            "qkv": array_t(*([0.0] * 4)),
            "attn_o": array_t(*([0.0] * 4)),
            "logits": array_t(*([0.0] * 4)),
        }
        ptrs = {
            name: _alloc(runtime, ctx, allocated, ctypes.sizeof(host))
            for name, host in hosts.items()
        }
        for name in ("a", "b", "c", "d", "q_weight", "o_weight"):
            _copy_to_device(runtime, ctx, ptrs[name], hosts[name], name)

        u32_2 = ctypes.c_uint32 * 2
        u32_3 = ctypes.c_uint32 * 3
        u32_8 = ctypes.c_uint32 * 8
        u32_11 = ctypes.c_uint32 * 11
        graph_hosts = {
            "tasks": make_tasks(ptrs),
            "dependents": u32_2(1, 2),
            "fanin": u32_3(0, 1, 1),
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
        state = make_state(plan, ptrs)
        _copy_to_device(runtime, ctx, ptrs["state"], state, "state")

        prepared = prepare_cuda_persistent_device_callable(
            artifact,
            grid_dim=plan["dag"]["scheduler_blocks"] + plan["dag"]["worker_blocks"],
            block_dim=plan["dag"]["block_dim"],
        )
        if runtime.prepare_callable(ctx, 0, prepared.byref()) != 0:
            raise RuntimeError("prepare_callable failed")
        callable_prepared = True

        args = CudaPersistentDagArgs(state=ptrs["state"])
        launch_records = []
        for iteration in range(repeat_runs):
            launch_state = {
                "fanin": u32_3(0, 1, 1),
                "ready_flags": u32_8(*([0] * 8)),
                "completion_flags": u32_8(*([0] * 8)),
                "counters": u32_11(*([0] * 11)),
                "scheduler_processed": (ctypes.c_uint32 * 1)(0),
                "qkv": array_t(*([0.0] * 4)),
                "attn_o": array_t(*([0.0] * 4)),
                "logits": array_t(*([0.0] * 4)),
            }
            for name, host in launch_state.items():
                _copy_to_device(runtime, ctx, ptrs[name], host, name)

            timing = PtoRunTiming()
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

            for name in ("qkv", "attn_o", "logits", "c", "d"):
                _copy_from_device(runtime, ctx, hosts[name], ptrs[name], name)
            for name in ("counters", "scheduler_processed"):
                _copy_from_device(runtime, ctx, launch_state[name], ptrs[name], name)

            observed = {
                "attention_qkv_out": [round(float(value), 6) for value in hosts["qkv"]],
                "attention_o_out": [round(float(value), 6) for value in hosts["attn_o"]],
                "logits_out": [round(float(value), 6) for value in hosts["logits"]],
                "c": [round(float(value), 6) for value in hosts["c"]],
                "d": [round(float(value), 6) for value in hosts["d"]],
            }
            iteration_expected = plan["decode_iterations"][iteration]["expected"]
            iteration_status, iteration_error = _status(
                expected=iteration_expected,
                observed=observed,
                counters=launch_state["counters"],
                scheduler_processed=launch_state["scheduler_processed"],
            )
            launch_records.append(
                {
                    "iteration": iteration,
                    "status": iteration_status,
                    "observed": observed,
                    "expected": iteration_expected,
                    "max_abs_error": iteration_error,
                    "scheduler_counters": {
                        "completed_count": int(launch_state["counters"][4]),
                        "error_count": int(launch_state["counters"][5]),
                        "error_code": int(launch_state["counters"][6]),
                        "error_task_id": int(launch_state["counters"][7]),
                        "scheduler_processed_count": int(launch_state["counters"][10]),
                        "scheduler_processed_by_block": [
                            int(launch_state["scheduler_processed"][0])
                        ],
                    },
                    "timing_ns": {
                        "host_wall": int(timing.host_wall_ns),
                        "device_wall": int(timing.device_wall_ns),
                    },
                }
            )
    finally:
        if callable_prepared:
            runtime.unregister_callable(ctx, 0)
        for ptr in reversed(allocated):
            runtime.device_free_ctx(ctx, ctypes.c_void_p(ptr))
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)

    final_record = launch_records[-1]
    max_abs_error = max(record["max_abs_error"] for record in launch_records)
    status = "pass" if all(record["status"] == "pass" for record in launch_records) else "fail"
    return {
        **plan,
        "kind": "pto_qwen_microdecode_live_execution",
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
        "observed": final_record["observed"],
        "max_abs_error": max_abs_error,
        "decode_loop_observations": launch_records,
        "decode_loop_summary": {
            "repeat_runs": repeat_runs,
            "total_completed_count": sum(
                record["scheduler_counters"]["completed_count"]
                for record in launch_records
            ),
            "total_error_count": sum(
                record["scheduler_counters"]["error_count"]
                for record in launch_records
            ),
            "total_scheduler_processed_count": sum(
                record["scheduler_counters"]["scheduler_processed_count"]
                for record in launch_records
            ),
        },
        "scheduler_counters": final_record["scheduler_counters"],
        "timing_ns": final_record["timing_ns"],
        "implemented_contracts": [
            *plan["implemented_contracts"],
            "controlled_proxy_live_microdecode_execution",
            "controlled_proxy_live_decode_loop_execution",
        ],
    }
