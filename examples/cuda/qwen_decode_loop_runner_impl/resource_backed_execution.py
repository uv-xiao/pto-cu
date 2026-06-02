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
    set_decode_step_state,
    task_shape_defaults,
    workspace_for_workload,
)
from qwen_decode_loop_runner_impl.logits_active_cols import (
    apply_logits_active_cols_override,
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
from qwen_decode_loop_runner_impl.workspace_pointers import (
    refresh_rope_tables_for_decode_position,
)
from qwen_persistent_proxy_live_impl.runtime import (
    PtoRunTiming,
    bind_persistent_runtime,
)
from qwen_persistent_task_bodies_impl.lifecycle import task_functions
from qwen_persistent_weight_materialization import build_materialization_manifest


ROOT = Path(__file__).resolve().parents[3]
PREFILL_READOUT_CALLABLES = {"qwen_final_norm", "qwen_logits"}


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
    max_task_count: int | None = None,
    task_selection: str = "prefix",
    worker_blocks: int = 1,
    logits_check_policy: str = "every_step",
    logits_active_cols: str | int | None = None,
    numeric_task_mode: str = "diagnostic",
    prefill_prompt: bool = False,
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
    descriptors = select_task_descriptors(
        descriptors,
        max_task_count=max_task_count,
        task_selection=task_selection,
    )
    descriptors, logits_active_cols_policy = apply_logits_active_cols_override(
        descriptors,
        logits_active_cols,
    )
    if not descriptors:
        return {"status": "not_run", "reason": "no_ready_descriptors"}
    scheduler_blocks = 1
    worker_blocks = max(1, int(worker_blocks))
    grid_dim = scheduler_blocks + worker_blocks
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
            grid_dim=grid_dim,
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
                numeric_task_mode=numeric_task_mode,
                prefill_prompt=prefill_prompt,
                scheduler_blocks=scheduler_blocks,
                block_dim=64,
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
        descriptors=descriptors,
        workload_results=workload_results,
        repeat_runs=repeat_runs,
        decode_step_limit=decode_step_limit,
        workload_ids=workload_ids,
        max_task_count=max_task_count,
        task_selection=task_selection,
        scheduler_blocks=scheduler_blocks,
        worker_blocks=worker_blocks,
        grid_dim=grid_dim,
        logits_check_policy=logits_check_policy,
        logits_active_cols_policy=logits_active_cols_policy,
        numeric_task_mode=numeric_task_mode,
        prefill_prompt=prefill_prompt,
        repo_relative=repo_relative,
    )


def select_task_descriptors(
    descriptors: list[dict[str, Any]],
    *,
    max_task_count: int | None,
    task_selection: str,
) -> list[dict[str, Any]]:
    if task_selection == "prefix":
        selected = descriptors
    elif task_selection == "first_layer_with_logits":
        selected = first_layer_with_logits_descriptors(descriptors)
    else:
        raise ValueError(f"unknown resource-backed task selection: {task_selection}")
    if max_task_count is not None:
        return selected[: max(1, int(max_task_count))]
    return selected


def first_layer_with_logits_descriptors(
    descriptors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in descriptors
        if (
            item.get("id") == "embedding_lookup"
            or str(item.get("id", "")).startswith("layer_0_")
            or item.get("id") in {"final_norm", "logits"}
        )
    ]


def prompt_prefill_descriptors(
    descriptors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in descriptors
        if item.get("callable") not in PREFILL_READOUT_CALLABLES
        and item.get("id") not in {"final_norm", "logits"}
    ]


def run_workload(
    *,
    session: Any,
    plan: dict[str, Any],
    descriptors: list[dict[str, Any]],
    activation_workspace: dict[str, Any],
    repeat_runs: int,
    decode_step_limit: int | None,
    logits_check_policy: str,
    numeric_task_mode: str,
    prefill_prompt: bool = False,
    scheduler_blocks: int = 1,
    block_dim: int = 64,
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
        numeric_task_mode=numeric_task_mode,
        task_shape_defaults=task_shape_defaults(plan),
    )
    if packet is None or workspace is None:
        return {"workload_id": plan["workload_id"], "status": "not_run"}
    final_callable = descriptors[-1].get("callable")

    prefill_results = []
    prefill_packet_len = 0
    prefill_task_policy = "not_requested"
    if prefill_prompt:
        prefill_task_policy = "omit_final_norm_and_logits_readout"
        prefill_items = prompt_prefill_descriptors(descriptors)
        prefill_packet = build_host_task_packet(
            descriptors=prefill_items,
            token_fields=token_fields,
            kv_fields=plan.get("kv_pointer_fields", {}),
            workspace=workspace,
            numeric_task_mode=numeric_task_mode,
            task_shape_defaults=task_shape_defaults(plan),
        )
        if prefill_packet is None:
            return {"workload_id": plan["workload_id"], "status": "not_run"}
        prefill_packet_len = len(prefill_packet)
        prefill_final_callable = (
            prefill_items[-1].get("callable") if prefill_items else None
        )
        prefill_count = max(int(plan.get("active_prompt_tokens", 0)), 0)
        for prefill_position in range(prefill_count):
            prefill_results.append(
                run_packet_once(
                    session=session,
                    packet=prefill_packet,
                    workspace=workspace,
                    final_callable=prefill_final_callable,
                    logits_check_policy=logits_check_policy,
                    scheduler_blocks=scheduler_blocks,
                    block_dim=block_dim,
                    decode_position=prefill_position,
                    decode_step_index=None,
                    phase="prompt_prefill",
                    prompt_stride=plan.get("runtime_prompt_tokens"),
                    token_fields=token_fields,
                )
            )

    execution_count = resource_backed_execution_count(
        plan=plan,
        repeat_runs=repeat_runs,
        decode_step_limit=decode_step_limit,
    )
    repeat_results = []
    for repeat_index in range(execution_count):
        step_index = repeat_index if decode_step_limit is not None else None
        decode_position = int(plan["first_decode_position"]) + int(step_index or 0)
        repeat_results.append(
            run_packet_once(
                session=session,
                packet=packet,
                workspace=workspace,
                final_callable=final_callable,
                logits_check_policy=logits_check_policy,
                scheduler_blocks=scheduler_blocks,
                block_dim=block_dim,
                decode_position=decode_position,
                decode_step_index=step_index,
                phase="decode",
                prompt_stride=plan.get("runtime_prompt_tokens"),
                token_fields=token_fields,
                repeat_index=repeat_index,
                execution_count=execution_count,
            )
        )
    return build_workload_result(
        plan=plan,
        packet_len=len(packet),
        repeat_results=repeat_results,
        decode_step_limit=decode_step_limit,
        logits_check_policy=logits_check_policy,
        numeric_task_mode=numeric_task_mode,
        prefill_packet_len=prefill_packet_len,
        prefill_task_policy=prefill_task_policy,
        prefill_results=prefill_results,
    )


def run_packet_once(
    *,
    session: Any,
    packet: Any,
    workspace: dict[str, Any],
    final_callable: str | None,
    logits_check_policy: str,
    scheduler_blocks: int,
    block_dim: int,
    decode_position: int,
    decode_step_index: int | None,
    phase: str,
    prompt_stride: int | None,
    token_fields: dict[str, dict[str, Any]],
    repeat_index: int = 0,
    execution_count: int = 1,
) -> dict[str, Any]:
    rope_table_refresh = refresh_rope_tables_for_decode_position(
        session.runtime,
        session.ctx,
        workspace,
        decode_position=decode_position,
    )
    set_decode_step_state(
        packet,
        step_index=decode_step_index,
        decode_position=decode_position,
    )
    graph = MaterializedGraph(
        session,
        packet,
        scheduler_blocks=scheduler_blocks,
        block_dim=block_dim,
    )
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
    if phase == "prompt_prefill":
        logits_summary = unchecked_logits_summary(
            policy=logits_check_policy,
            repeat_index=repeat_index,
            execution_count=execution_count,
            reason="prompt_prefill_logits_not_checked",
        )
    elif final_callable != "qwen_logits":
        logits_summary = unchecked_logits_summary(
            policy=logits_check_policy,
            repeat_index=repeat_index,
            execution_count=execution_count,
            reason="bounded_prefix_without_logits_task",
        )
    elif should_check_logits(
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
    decode_feedback = {"status": "not_requested"}
    if phase != "prompt_prefill":
        decode_feedback = apply_decode_feedback(
            session=session,
            token_fields=token_fields,
            decode_step_index=decode_step_index,
            decode_position=decode_position,
            prompt_stride=prompt_stride,
            logits_summary=logits_summary,
            device_committed=True,
        )
    return {
        "repeat_index": repeat_index,
        "decode_step_index": decode_step_index,
        "decode_position": int(decode_position),
        "phase": phase,
        "status": (
            "pass"
            if status == 0
            and counters["completed_count"] == len(packet)
            and counters["error_count"] == 0
            else "fail"
        ),
        "run_prepared_status": int(status),
        "rope_table_refresh": rope_table_refresh,
        "scheduler_counters": counters,
        "output_sample": graph.read_output_sample(workspace),
        "logits_summary": logits_summary,
        "decode_feedback": decode_feedback,
        "timing_ns": {
            "host_wall": int(timing.host_wall_ns),
            "device_wall": int(timing.device_wall_ns),
        },
    }
