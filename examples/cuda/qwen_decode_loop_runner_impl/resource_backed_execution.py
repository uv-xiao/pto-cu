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

from qwen_decode_loop_runner_impl.launch_preflight import (
    build_host_task_packet,
    keyed_fields,
    workspace_for_workload,
)
from qwen_decode_loop_runner_impl.resource_execution_policy import (
    decode_step_execution_summary,
    implemented_contracts,
    resource_backed_execution_count,
)
from qwen_decode_loop_runner_impl.resource_graph import MaterializedGraph
from qwen_persistent_proxy_live_impl.runtime import (
    PtoRunTiming,
    bind_persistent_runtime,
    device_name,
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
            )
            for plan in plans
        ]
    except Exception as exc:  # noqa: BLE001 - artifact should capture failure.
        return {"status": "fail", "reason": type(exc).__name__, "message": str(exc)}
    finally:
        if callable_prepared:
            runtime.unregister_callable(ctx, 0)

    passed = workload_results and all(
        item["status"] == "pass" for item in workload_results
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_resource_backed_execution",
        "status": "pass" if passed else "fail",
        "runtime": "cuda/persistent_device",
        "serving_coverage": "diagnostic_resource_backed_qwen_dag",
        "device": {
            "ordinal": session.device,
            "name": device_name(session.device),
            "arch": arch,
        },
        "artifact": artifact_summary(prepared),
        "context_policy": "one_cuda_context_for_all_resource_owners",
        "repeat_policy": {
            "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
            "repeat_runs_per_workload": max(1, int(repeat_runs)),
            "decode_step_limit": decode_step_limit,
            "graph_state_policy": "fresh_graph_state_per_repeat",
        },
        "decode_step_execution": decode_step_execution_summary(
            workload_results,
            decode_step_limit=decode_step_limit,
        ),
        "workloads": workload_results,
        "implemented_contracts": implemented_contracts(decode_step_limit),
        "remaining_runtime_gaps": [
            "full_qwen_numerical_correctness",
            "full_serving_viewer_result_import",
        ],
    }


def run_workload(
    *,
    session: Any,
    plan: dict[str, Any],
    descriptors: list[dict[str, Any]],
    activation_workspace: dict[str, Any],
    repeat_runs: int,
    decode_step_limit: int | None,
) -> dict[str, Any]:
    workspace = workspace_for_workload(
        activation_workspace=activation_workspace,
        workload_id=plan["workload_id"],
        task_count=len(descriptors),
    )
    packet = build_host_task_packet(
        descriptors=descriptors,
        token_fields=keyed_fields(plan.get("token_pointer_fields", [])),
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
        repeat_results.append(
            {
                "repeat_index": repeat_index,
                "decode_step_index": (
                    repeat_index if decode_step_limit is not None else None
                ),
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
                "logits_summary": graph.read_logits_summary(workspace),
                "timing_ns": {
                    "host_wall": int(timing.host_wall_ns),
                    "device_wall": int(timing.device_wall_ns),
                },
            }
        )
    last = repeat_results[-1]
    total_host_wall = sum(
        item["timing_ns"]["host_wall"] for item in repeat_results
    )
    total_device_wall = sum(
        item["timing_ns"]["device_wall"] for item in repeat_results
    )
    result = {
        "workload_id": plan["workload_id"],
        "status": (
            "pass"
            if all(item["status"] == "pass" for item in repeat_results)
            else "fail"
        ),
        "run_prepared_status": int(last["run_prepared_status"]),
        "repeat_runs": len(repeat_results),
        "planned_decode_steps": int(plan["decode_steps"]),
        "executed_decode_steps": (
            len(repeat_results) if decode_step_limit is not None else 0
        ),
        "decode_step_limit": decode_step_limit,
        "execution_mode": (
            "bounded_decode_steps"
            if decode_step_limit is not None
            else "repeat_submissions"
        ),
        "repeat_results": repeat_results,
        "graph_task_count": len(packet),
        "scheduler_counters": last["scheduler_counters"],
        "total_completed_count": sum(
            item["scheduler_counters"]["completed_count"] for item in repeat_results
        ),
        "total_error_count": sum(
            item["scheduler_counters"]["error_count"] for item in repeat_results
        ),
        "output_sample": last["output_sample"],
        "logits_summary": last["logits_summary"],
        "logits_summary_stable": logits_summary_stable(repeat_results),
        "timing_ns": {
            "host_wall": total_host_wall,
            "device_wall": total_device_wall,
        },
    }
    return result


def logits_summary_stable(repeat_results: list[dict[str, Any]]) -> bool:
    if not repeat_results:
        return False
    first = repeat_results[0].get("logits_summary", {})
    first_key = (
        first.get("sample_checksum"),
        first.get("topk", [{}])[0].get("token_id") if first.get("topk") else None,
        first.get("written_element_count"),
        first.get("sampled_element_count"),
    )
    for item in repeat_results[1:]:
        summary = item.get("logits_summary", {})
        key = (
            summary.get("sample_checksum"),
            summary.get("topk", [{}])[0].get("token_id")
            if summary.get("topk")
            else None,
            summary.get("written_element_count"),
            summary.get("sampled_element_count"),
        )
        if key != first_key:
            return False
    return True


def artifact_summary(prepared: Any) -> dict[str, Any]:
    artifact = prepared.artifact
    return {
        "cache_key": artifact.cache_key,
        "cache_hit": artifact.cache_hit,
        "source_path": repo_relative(Path(artifact.source_path)),
        "ptx_path": repo_relative(Path(artifact.ptx_path)),
        "entry_name": artifact.entry_name,
        "source_kind": artifact.source_kind,
    }
