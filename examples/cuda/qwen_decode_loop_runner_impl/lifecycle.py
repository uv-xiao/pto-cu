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
from qwen_persistent_weight_materialization import build_materialization_manifest


ROOT = Path(__file__).resolve().parents[3]

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def ready_resident_descriptors(
    resident_lifecycle: dict[str, Any],
) -> list[dict[str, Any]]:
    materialization = build_materialization_manifest(
        pointer_table=resident_lifecycle.get("pointer_table"),
    )
    return [
        item
        for item in materialization.get("materialized_task_descriptors", [])
        if item.get("status") == "ready"
    ]


def resource_lifecycle_policies(
    *,
    kv_lifecycle: dict[str, Any],
) -> dict[str, Any]:
    kv_table = kv_lifecycle.get("pointer_table", {})
    return {
        "kv_cache": {
            "allocation_policy": kv_table.get("allocation_policy"),
            "initialization_policy": kv_table.get("initialization_policy"),
        },
    }


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
    resource_backed_batch_size: int | None = None,
    resource_backed_max_tasks: int | None = None,
    resource_backed_task_selection: str = "prefix",
    resource_backed_layer_count: int | None = None,
    resource_backed_worker_blocks: int = 1,
    resource_backed_logits_check_policy: str = "every_step",
    resource_backed_logits_active_cols: str | int | None = None,
    resource_backed_projection_active_cols: str | int | None = None,
    resource_backed_activation_sample_columns: str | None = None,
    resource_backed_activation_row_dump_descriptor_ids: str | None = None,
    resource_backed_numeric_task_mode: str = "diagnostic",
    resource_backed_prefill_prompt: bool = False,
    resource_backed_progress_callback: Any | None = None,
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
            workload_ids=resource_backed_workloads,
            batch_size=resource_backed_batch_size,
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
    resident_descriptors = ready_resident_descriptors(resident_lifecycle)
    graph_task_count = len(resident_descriptors) or int(
        resident_lifecycle.get("materialized_task_count", 0),
    )
    activation_workspace = (
        session.open_activation_workspace(
            plans=plans,
            graph_task_count=graph_task_count,
            descriptors=resident_descriptors,
        )
        if session is not None
        else build_activation_workspace_lifecycle(
            plans=plans,
            graph_task_count=graph_task_count,
            descriptors=resident_descriptors,
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
                max_task_count=resource_backed_max_tasks,
                task_selection=resource_backed_task_selection,
                layer_count=resource_backed_layer_count,
                worker_blocks=resource_backed_worker_blocks,
                logits_check_policy=resource_backed_logits_check_policy,
                logits_active_cols=resource_backed_logits_active_cols,
                projection_active_cols=resource_backed_projection_active_cols,
                activation_sample_columns=(
                    resource_backed_activation_sample_columns
                ),
                activation_row_dump_descriptor_ids=(
                    resource_backed_activation_row_dump_descriptor_ids
                ),
                numeric_task_mode=resource_backed_numeric_task_mode,
                prefill_prompt=resource_backed_prefill_prompt,
                progress_callback=resource_backed_progress_callback,
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
            decode_step_status = resource_backed_execution.get(
                "decode_step_execution",
                {},
            ).get("status")
            if decode_step_status in {
                "bounded_decode_steps_executed",
                "policy_length_decode_steps_executed",
            }:
                implemented_contracts.append(
                    "qwen_resource_backed_decode_step_execution",
                )
            if decode_step_status == "policy_length_decode_steps_executed":
                implemented_contracts.append(
                    "qwen_resource_backed_policy_length_decode_execution",
                )
            resource_contracts = resource_backed_execution.get(
                "implemented_contracts",
                [],
            )
            for contract in [
                "qwen_diagnostic_decode_token_feedback",
                "qwen_device_decode_token_feedback",
                "qwen_resource_backed_unit_numeric_task_mode",
                "qwen_resource_backed_external_rmsnorm_scale",
                "qwen_resource_backed_full_rmsnorm_reduction",
                "qwen_resource_backed_weighted_elementwise_branches",
                "qwen_dynamic_rope_table_refresh",
                "qwen_resource_backed_policy_length_decode_execution",
            ]:
                if contract in resource_contracts:
                    implemented_contracts.append(contract)
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
    workload_preflights = [
        item.get("launch_packet_preflight", {})
        for item in graph_materialization.get("workloads", [])
    ]
    if all(
        item.get("launch_packet_preflight", {}).get("status")
        == "resource_backed_launch_packet_workspace_bound"
        for item in graph_materialization.get("workloads", [])
    ):
        implemented_contracts.append("qwen_activation_workspace_launch_packet_binding")
    runtime_buffer_role_sets = [
        set(item.get("workspace_pointer_policy", {}).get("runtime_buffers", {}))
        for item in workload_preflights
    ]
    if runtime_buffer_role_sets and all(
        {"rope_cos_table", "rope_sin_table"} <= roles
        for roles in runtime_buffer_role_sets
    ):
        implemented_contracts.append("qwen_rope_table_launch_packet_binding")
    if runtime_buffer_role_sets and all(
        "kv_page_table" in roles for roles in runtime_buffer_role_sets
    ):
        implemented_contracts.append("qwen_kv_page_table_launch_packet_binding")
    workspace_plans = activation_workspace.get("workspace_plans", [])
    if workspace_plans and all(
        item.get("rope_table_policy")
        == "position_correct_for_first_decode_position"
        for item in workspace_plans
    ):
        implemented_contracts.append("qwen_position_rope_table_population")
    remaining_runtime_gaps = [
        "numerically_correct_qwen_kernel_bodies",
        "viewer_result_import",
    ]
    if "qwen_resource_backed_policy_length_decode_execution" not in implemented_contracts:
        remaining_runtime_gaps.insert(1, "full_cuda_live_decode_loop_execution")
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
        "resource_lifecycle_policies": resource_lifecycle_policies(
            kv_lifecycle=kv_lifecycle,
        ),
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
        "remaining_runtime_gaps": remaining_runtime_gaps,
    }
