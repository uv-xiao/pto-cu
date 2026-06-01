"""Run diagnostic Qwen DAG packets against live resource-backed pointers."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagArgs,
    compile_cuda_persistent_device,
    prepare_cuda_persistent_device_callable,
)

from qwen_decode_loop_runner_impl.decode_feedback import (
    apply_decode_feedback,
)
from qwen_decode_loop_runner_impl.launch_preflight import (
    build_host_task_packet,
    keyed_fields,
    set_decode_step_index,
    workspace_for_workload,
)
from qwen_decode_loop_runner_impl.resource_execution_policy import (
    resource_backed_execution_count,
)
from qwen_decode_loop_runner_impl.resource_check_policy import (
    normalize_logits_check_policy,
    select_workload_plans,
    should_check_logits,
    unchecked_logits_summary,
)
from qwen_decode_loop_runner_impl.resource_backed_results import (
    build_execution_result,
    build_workload_result,
)
from qwen_decode_loop_runner_impl.resource_graph import MaterializedGraph
from qwen_persistent_proxy_live_impl.runtime import (
    PtoRunTiming,
    bind_persistent_runtime,
)
from qwen_persistent_task_bodies_impl.lifecycle import task_functions
from qwen_persistent_weight_materialization import build_materialization_manifest


ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run_resource_backed_execution(
    *,
    session: Any,
    plans: list[dict[str, Any]],
    resident_lifecycle: dict[str, Any],
    activation_workspace: dict[str, Any],
    arch: str,
    cache_root: Path | None,
    repeat_runs: int = 1,
    decode_step_limit: int | None = None,
    workload_ids: list[str] | None = None,
    logits_check_policy: str = "every_step",
) -> dict[str, Any]:
    runtime = session.runtime
    ctx = session.ctx
    if runtime is None or ctx is None:
        return {"status": "not_run", "reason": "session_not_open"}
    bind_persistent_runtime(runtime)
    materialization = build_materialization_manifest(
        pointer_table=resident_lifecycle.get("pointer_table"),
    )
    descriptors = [
        item
        for item in materialization.get("materialized_task_descriptors", [])
        if item.get("status") == "ready"
    ]
    if not descriptors:
        return {"status": "not_run", "reason": "no_ready_descriptors"}
    logits_check_policy = normalize_logits_check_policy(logits_check_policy)
    selected_plans = select_workload_plans(plans, workload_ids)
    if not selected_plans:
        return {
            "status": "not_run",
            "reason": "no_matching_resource_backed_workloads",
            "requested_workloads": workload_ids or [],
        }

    prepared = None
    callable_prepared = False
    try:
        artifact = compile_cuda_persistent_device(
            task_functions(),
            arch=arch,
            cache_root=cache_root,
        )
        prepared = prepare_cuda_persistent_device_callable(
            artifact,
            grid_dim=2,
            block_dim=64,
        )
        if runtime.prepare_callable(ctx, 0, prepared.byref()) != 0:
            return {"status": "fail", "reason": "prepare_callable_failed"}
        callable_prepared = True
        workload_results = [
            run_workload(
                session=session,
                plan=plan,
                descriptors=descriptors,
                activation_workspace=activation_workspace,
                repeat_runs=repeat_runs,
                decode_step_limit=decode_step_limit,
                logits_check_policy=logits_check_policy,
            )
            for plan in selected_plans
        ]
    except Exception as exc:  # noqa: BLE001 - artifact should capture failure.
        return {"status": "fail", "reason": type(exc).__name__, "message": str(exc)}
    finally:
        if callable_prepared:
            runtime.unregister_callable(ctx, 0)

    return build_execution_result(
        session=session,
        arch=arch,
        prepared=prepared,
        workload_results=workload_results,
        repeat_runs=repeat_runs,
        decode_step_limit=decode_step_limit,
        workload_ids=workload_ids,
        logits_check_policy=logits_check_policy,
        repo_relative=repo_relative,
    )


def run_workload(
    *,
    session: Any,
    plan: dict[str, Any],
    descriptors: list[dict[str, Any]],
    activation_workspace: dict[str, Any],
    repeat_runs: int,
    decode_step_limit: int | None,
    logits_check_policy: str,
) -> dict[str, Any]:
    workspace = workspace_for_workload(
        activation_workspace=activation_workspace,
        workload_id=plan["workload_id"],
        task_count=len(descriptors),
    )
    token_fields = keyed_fields(plan.get("token_pointer_fields", []))
    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=token_fields,
        kv_fields=plan.get("kv_pointer_fields", {}),
        workspace=workspace,
    )
    if packet is None or workspace is None:
        return {"workload_id": plan["workload_id"], "status": "not_run"}

    execution_count = resource_backed_execution_count(
        plan=plan,
        repeat_runs=repeat_runs,
        decode_step_limit=decode_step_limit,
    )
    repeat_results = []
    for repeat_index in range(execution_count):
        step_index = repeat_index if decode_step_limit is not None else None
        set_decode_step_index(packet, step_index)
        graph = MaterializedGraph(session, packet)
        timing = PtoRunTiming()
        args = CudaPersistentDagArgs(state=graph.ptrs["state"])
        status = session.runtime.run_prepared(
            session.ctx,
            None,
            0,
            ctypes.byref(args),
            graph.block_dim,
            0,
            0,
            0,
            0,
            0,
            None,
            ctypes.byref(timing),
        )
        counters = graph.read_counters()
        if should_check_logits(
            policy=logits_check_policy,
            repeat_index=repeat_index,
            execution_count=execution_count,
        ):
            logits_summary = graph.read_logits_summary(workspace)
        else:
            logits_summary = unchecked_logits_summary(
                policy=logits_check_policy,
                repeat_index=repeat_index,
                execution_count=execution_count,
            )
        decode_feedback = apply_decode_feedback(
            session=session,
            token_fields=token_fields,
            decode_step_index=step_index,
            logits_summary=logits_summary,
            device_committed=True,
        )
        repeat_results.append(
            {
                "repeat_index": repeat_index,
                "decode_step_index": step_index,
                "status": (
                    "pass"
                    if status == 0
                    and counters["completed_count"] == len(packet)
                    and counters["error_count"] == 0
                    else "fail"
                ),
                "run_prepared_status": int(status),
                "scheduler_counters": counters,
                "output_sample": graph.read_output_sample(workspace),
                "logits_summary": logits_summary,
                "decode_feedback": decode_feedback,
                "timing_ns": {
                    "host_wall": int(timing.host_wall_ns),
                    "device_wall": int(timing.device_wall_ns),
                },
            }
        )
    return build_workload_result(
        plan=plan,
        packet_len=len(packet),
        repeat_results=repeat_results,
        decode_step_limit=decode_step_limit,
        logits_check_policy=logits_check_policy,
    )
