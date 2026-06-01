"""Compose Qwen resource owners into decode-loop submission plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_decode_loop_runner_impl.activation_workspace import (
    build_activation_workspace_lifecycle,
)
from qwen_decode_loop_runner_impl.bridge_contracts import (
    cuda_live_bridge_contract,
    unit_math_live_bridge_contract,
)
from qwen_decode_loop_runner_impl.graph_materialization import (
    graph_materialization_contract,
)
from qwen_decode_loop_runner_impl.resources import build_resources
from qwen_decode_loop_runner_impl.resource_backed_execution import (
    run_resource_backed_execution,
)
from qwen_decode_loop_runner_impl.single_context_session import (
    open_single_context_live_session,
)
from qwen_decode_loop_runner_impl.submission_plan import build_submission_plans
from qwen_decode_loop_runner_impl.submission import submission_descriptor_contract


ROOT = Path(__file__).resolve().parents[3]

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_decode_loop_runner(
    *,
    mode: str = "offline",
    cache_dir: Path | None = None,
    unit_math_live_payload: dict[str, Any] | None = None,
    token_cuda_live: bool = False,
    kv_cuda_live: bool = False,
    resident_cuda_live: bool = False,
    device: int = 0,
    host_runtime: Path | None = None,
    submission_smoke_payload: dict[str, Any] | None = None,
    workspace_cuda_live: bool = False,
    single_context_live_session: bool = False,
    run_resource_backed_smoke: bool = False,
    resource_backed_repeat_runs: int = 1,
    resource_backed_decode_steps: int | None = None,
    resource_backed_workloads: list[str] | None = None,
    resource_backed_logits_check_policy: str = "every_step",
    arch: str = "compute_80",
) -> dict[str, Any]:
    session_payload: dict[str, Any] | None = None
    resource_backed_execution: dict[str, Any] | None = None
    if single_context_live_session:
        session = open_single_context_live_session(
            mode=mode,
            cache_dir=cache_dir,
            device=device,
            host_runtime=host_runtime,
        )
        token_lifecycle = session.token_lifecycle()
        kv_lifecycle = session.kv_lifecycle()
        resident_lifecycle = session.resident_lifecycle()
    else:
        session = None
        token_lifecycle, kv_lifecycle, resident_lifecycle = build_resources(
            mode=mode,
            cache_dir=cache_dir,
            token_cuda_live=token_cuda_live,
            kv_cuda_live=kv_cuda_live,
            resident_cuda_live=resident_cuda_live,
            device=device,
            host_runtime=host_runtime,
        )
    plans = build_submission_plans(
        token_lifecycle=token_lifecycle,
        kv_lifecycle=kv_lifecycle,
        resident_lifecycle=resident_lifecycle,
    )
    graph_task_count = int(resident_lifecycle.get("materialized_task_count", 0))
    activation_workspace = (
        session.open_activation_workspace(
            plans=plans,
            graph_task_count=graph_task_count,
        )
        if session is not None
        else build_activation_workspace_lifecycle(
            plans=plans,
            graph_task_count=graph_task_count,
            cuda_live=workspace_cuda_live,
            device=device,
            host_runtime=host_runtime,
        )
    )
    resource_modes = {
        "token_pointer_table": token_lifecycle.get("mode", "unknown"),
        "kv_cache": kv_lifecycle.get("mode", "unknown"),
        "resident_weight_table": resident_lifecycle.get("mode", "unknown"),
        "activation_workspace": activation_workspace.get("mode", "unknown"),
    }
    live_owners = [
        name
        for name, resource_mode in resource_modes.items()
        if resource_mode == "cuda_live"
    ]
    implemented_contracts = [
        "decode_loop_owner_lifetime_order",
        "persistent_dag_submission_plan",
        "output_token_accounting_plan",
        "cuda_live_resource_bridge_contract",
        "qwen_unit_math_live_bridge_contract",
    ]
    if token_lifecycle.get("mode") == "cuda_live":
        implemented_contracts.append("cuda_live_token_pointer_table_in_runner")
    if kv_lifecycle.get("mode") == "cuda_live":
        implemented_contracts.append("cuda_live_kv_cache_owner_in_runner")
    if resident_lifecycle.get("mode") == "cuda_live":
        implemented_contracts.append("cuda_live_resident_weight_table_in_runner")
    if activation_workspace.get("mode") == "cuda_live":
        implemented_contracts.append("cuda_live_activation_workspace_in_runner")
    resource_status = {
        "token_pointer_table": token_lifecycle.get("status"),
        "kv_cache": kv_lifecycle.get("status"),
        "resident_weight_table": resident_lifecycle.get("status"),
        "activation_workspace": activation_workspace.get("status"),
    }
    graph_materialization = graph_materialization_contract(
        plans=plans,
        resident_lifecycle=resident_lifecycle,
        activation_workspace=activation_workspace,
    )
    if run_resource_backed_smoke:
        resource_backed_execution = (
            run_resource_backed_execution(
                session=session,
                plans=plans,
                resident_lifecycle=resident_lifecycle,
                activation_workspace=activation_workspace,
                arch=arch,
                cache_root=cache_dir,
                repeat_runs=resource_backed_repeat_runs,
                decode_step_limit=resource_backed_decode_steps,
                workload_ids=resource_backed_workloads,
                logits_check_policy=resource_backed_logits_check_policy,
            )
            if session is not None
            else {
                "status": "not_run",
                "reason": "single_context_live_session_required",
            }
        )
    if session is not None:
        close_summary = session.close()
        activation_workspace["pointer_table"] = session.closed_table(
            activation_workspace["pointer_table"],
            "activation_workspace",
        )
        token_lifecycle = session.token_lifecycle()
        kv_lifecycle = session.kv_lifecycle()
        resident_lifecycle = session.resident_lifecycle()
        session_payload = {
            "status": "single_context_launch_packet_session_ready",
            "runtime": "cuda/persistent_device",
            "context_policy": "one_cuda_context_for_all_resource_owners",
            "graph_materialized_before_close": (
                graph_materialization.get("status")
                == "resource_backed_graph_materialized"
            ),
            "launch_packet_bound_before_close": all(
                item.get("launch_packet_preflight", {}).get("status")
                == "resource_backed_launch_packet_workspace_bound"
                for item in graph_materialization.get("workloads", [])
            ),
            "close_summary": close_summary,
        }
        resource_modes = {
            "token_pointer_table": token_lifecycle.get("mode", "unknown"),
            "kv_cache": kv_lifecycle.get("mode", "unknown"),
            "resident_weight_table": resident_lifecycle.get("mode", "unknown"),
            "activation_workspace": activation_workspace.get("mode", "unknown"),
        }
        resource_status = {
            "token_pointer_table": token_lifecycle.get("status"),
            "kv_cache": kv_lifecycle.get("status"),
            "resident_weight_table": resident_lifecycle.get("status"),
            "activation_workspace": activation_workspace.get("status"),
        }
        implemented_contracts.append("single_context_live_resource_session")
        if resource_backed_execution and resource_backed_execution.get("status") == "pass":
            implemented_contracts.append("qwen_resource_backed_diagnostic_execution")
            if resource_backed_execution.get("decode_step_execution", {}).get(
                "status",
            ) == "bounded_decode_steps_executed":
                implemented_contracts.append(
                    "qwen_resource_backed_decode_step_execution",
                )
            if (
                "qwen_diagnostic_decode_token_feedback"
                in resource_backed_execution.get("implemented_contracts", [])
            ):
                implemented_contracts.append("qwen_diagnostic_decode_token_feedback")
            if (
                "qwen_device_decode_token_feedback"
                in resource_backed_execution.get("implemented_contracts", [])
            ):
                implemented_contracts.append("qwen_device_decode_token_feedback")
    submission_descriptors = submission_descriptor_contract(
        plans=plans,
        resource_modes=resource_modes,
        resource_status={
            "token_pointer_table": resource_status["token_pointer_table"],
            "kv_cache": resource_status["kv_cache"],
            "resident_weight_table": resource_status["resident_weight_table"],
        },
        execution_evidence=submission_smoke_payload,
    )
    if submission_descriptors["status"] == "resource_backed_descriptors_ready":
        implemented_contracts.append("qwen_decode_loop_submission_descriptors")
    if submission_smoke_payload is not None:
        implemented_contracts.append("qwen_decode_loop_submission_smoke_execution")
    if graph_materialization["status"] == "resource_backed_graph_materialized":
        implemented_contracts.append("qwen_resource_backed_graph_materialization")
    if all(
        item.get("launch_packet_preflight", {}).get("status")
        in {
            "resource_backed_launch_packet_preflight_ready",
            "resource_backed_launch_packet_workspace_bound",
        }
        for item in graph_materialization.get("workloads", [])
    ):
        implemented_contracts.append("qwen_resource_backed_launch_packet_preflight")
    if all(
        item.get("launch_packet_preflight", {}).get("status")
        == "resource_backed_launch_packet_workspace_bound"
        for item in graph_materialization.get("workloads", [])
    ):
        implemented_contracts.append("qwen_activation_workspace_launch_packet_binding")
    return {
        "schema_version": 1,
        "kind": "pto_qwen_decode_loop_runner",
        "status": "decode_loop_runner_plan_ready",
        "mode": (
            "partial_cuda_live_submission_plan"
            if live_owners
            else "dry_run_submission_plan"
        ),
        "resource_lifecycle_status": resource_status,
        "resource_lifecycle_modes": resource_modes,
        "cuda_live_resource_owners": live_owners,
        "activation_workspace_lifecycle": activation_workspace,
        "single_context_live_session": session_payload,
        "resource_backed_execution": resource_backed_execution,
        "cuda_live_submission_descriptor_contract": submission_descriptors,
        "resource_backed_graph_materialization": graph_materialization,
        "cuda_live_bridge_contract": cuda_live_bridge_contract(),
        "unit_math_live_bridge_contract": unit_math_live_bridge_contract(
            unit_math_live_payload,
        ),
        "dag_submission_plans": plans,
        "total_decode_iterations": sum(item["decode_steps"] for item in plans),
        "implemented_contracts": implemented_contracts,
        "remaining_runtime_gaps": [
            "numerically_correct_qwen_kernel_bodies",
            "full_cuda_live_decode_loop_execution",
            "viewer_result_import",
        ],
    }
