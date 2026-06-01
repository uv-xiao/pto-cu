"""Qwen decode-loop activation and logits workspace ownership."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from qwen_decode_loop_runner_impl.workspace_pointers import (
    dry_run_workspace_table,
    live_workspace_table,
)


ROOT = Path(__file__).resolve().parents[3]
LIFECYCLE_PLAN = ROOT / "examples" / "cuda" / "qwen_serving_lifecycle_plan.py"
DEFAULT_HOST_RUNTIME = (
    ROOT
    / "build"
    / "lib"
    / "cuda"
    / "onboard"
    / "persistent_device"
    / "libhost_runtime.so"
)


def load_model_shape() -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(
        "qwen_lifecycle_for_activation_workspace",
        LIFECYCLE_PLAN,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {LIFECYCLE_PLAN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_lifecycle_plan()["model_shape"]


def build_activation_workspace_lifecycle(
    *,
    plans: list[dict[str, Any]],
    graph_task_count: int,
    cuda_live: bool = False,
    device: int = 0,
    host_runtime: Path | None = None,
) -> dict[str, Any]:
    model_shape = load_model_shape()
    workspace_plans = [
        workspace_plan(
            plan=plan,
            graph_task_count=graph_task_count,
            model_shape=model_shape,
        )
        for plan in plans
    ]
    pointer_table = (
        live_workspace_table(
            workspace_plans=workspace_plans,
            device=device,
            host_runtime=host_runtime or DEFAULT_HOST_RUNTIME,
        )
        if cuda_live
        else dry_run_workspace_table(workspace_plans=workspace_plans)
    )
    ready = (
        pointer_table.get("status") == "activation_workspace_pointer_table_ready"
        and pointer_table.get("mode") == "cuda_live"
        and pointer_table.get("freed_pointer_count")
        == pointer_table.get("pointer_count")
    )
    return {
        "schema_version": 1,
        "kind": "pto_qwen_activation_workspace_lifecycle",
        "status": (
            "activation_workspace_lifecycle_ready"
            if ready
            else "activation_workspace_lifecycle_planned"
        ),
        "mode": "cuda_live" if cuda_live else "dry_run_pointer_lifecycle",
        "runtime": "cuda/persistent_device",
        "model_id": model_shape["model_id"],
        "hidden_size": model_shape["hidden_size"],
        "vocab_size": model_shape["vocab_size"],
        "element_dtype": "float32",
        "workspace_plans": workspace_plans,
        "pointer_table": pointer_table,
        "implemented_contracts": [
            "activation_workspace_lifetime_owner",
            "float_logits_or_sampling_output_workspace",
            "cuda_live_activation_workspace" if cuda_live else "dry_run_workspace_plan",
        ],
        "remaining_runtime_gaps": [
            "attach_workspace_to_run_prepared_state",
            "numerically_correct_qwen_kernel_bodies",
            "cuda_live_decode_loop_execution",
        ],
    }


def workspace_plan(
    *,
    plan: dict[str, Any],
    graph_task_count: int,
    model_shape: dict[str, Any],
) -> dict[str, Any]:
    hidden_elements = int(plan["max_batch_size"]) * int(model_shape["hidden_size"])
    logits_elements = int(plan["max_batch_size"]) * int(model_shape["vocab_size"])
    activation_count = max(graph_task_count - 1, 0)
    return {
        "workload_id": plan["workload_id"],
        "status": "activation_workspace_plan_ready",
        "max_batch_size": int(plan["max_batch_size"]),
        "graph_task_count": graph_task_count,
        "activation_buffer_count": activation_count,
        "activation_buffer_elements": hidden_elements,
        "activation_buffer_bytes": hidden_elements * 4,
        "logits_buffer_count": 1,
        "logits_buffer_elements": logits_elements,
        "logits_buffer_bytes": logits_elements * 4,
        "total_buffer_count": activation_count + 1,
        "total_byte_count": activation_count * hidden_elements * 4
        + logits_elements * 4,
    }
