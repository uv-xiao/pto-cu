#!/usr/bin/env python3
"""Run or describe a persistent-device MoE dispatch/combine graph."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagArgs,
    CudaPersistentDagState,
    CudaPersistentDagTask,
    CudaPersistentTaskBodyFunction,
    CudaTaskBody,
    prepare_cuda_persistent_device_callable,
    render_persistent_dag_source,
)
from simpler_setup.cuda_preflight import cuda_skip_reason
from simpler_setup.gluon_gen import generate_gluon_persistent_task_body
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.runtime_builder import RuntimeBuilder


DAG_SHAPE = "graph_descriptor_moe_dispatch_combine"
DEFAULT_OUTPUT_JSON = None
TWO_DEVICE_EVIDENCE_SCOPE = "same-node-two-device-baseline"
HANDOFF_SCOPE = "persistent-moe-plus-nccl-worker-control"
UCCL_EP_HANDOFF_SCOPE = "persistent-moe-plus-uccl-ep-adapter"
UCCL_EP_FUSED_BOUNDARY_SCOPE = "reduced-fused-cross-gpu-expert-parallel-moe-boundary"
TWO_DEVICE_EVIDENCE_STATEMENT = (
    "same-node two-device baseline evidence: the existing persistent MoE "
    "dispatch/combine graph ran independently on each requested CUDA device; "
    "this is not fused cross-GPU expert-parallel MoE"
)
HANDOFF_EVIDENCE_STATEMENT = (
    "same-node two-device handoff gate: the existing persistent MoE graph ran "
    "on both requested devices, then the descriptor-backed NCCL worker-control "
    "operation path ran on the same device ids"
)
UCCL_EP_HANDOFF_EVIDENCE_STATEMENT = (
    "same-node two-device UCCL-EP handoff gate: the existing persistent MoE "
    "graph ran on both requested devices, then the Python-side UCCL-EP "
    "dispatch/combine adapter ran on the same device ids"
)
UCCL_EP_FUSED_BOUNDARY_EVIDENCE_STATEMENT = (
    "structured unsupported boundary: the accepted persistent MoE plus "
    "UCCL-EP adapter handoff can be run in one command, but this is "
    "non-evidence for fused cross-GPU expert-parallel MoE until the "
    "persistent_device_uccl_ep_runtime_fusion boundary exists"
)
UCCL_EP_FUSED_BOUNDARY_MISSING = (
    "persistent_device_uccl_ep_runtime_fusion",
    "shared_dispatch_combine_payload_between_persistent_graph_and_uccl_ep",
    "device_side_cross_gpu_expert_parallel_routing",
)
RUNTIME_FUSION_PRODUCER = "persistent_device_uccl_ep_runtime_fusion"
RUNTIME_FUSION_LIFETIME_SEQUENCE = (
    ("allocated", "persistent_device_graph"),
    ("dispatch_ready", "persistent_device_graph"),
    ("dispatch_in_flight", "uccl_ep_runtime"),
    ("combine_ready", "persistent_device_graph"),
    ("combine_in_flight", "uccl_ep_runtime"),
    ("complete", "persistent_device_graph"),
    ("released", "released"),
)
NO_SHARED_PAYLOAD_OWNERSHIP_REASON = (
    "runtime component has not created or transferred shared payload ownership"
)
CONTEXT_DEFINITION = """
struct PtoTaskContext {
    const PtoCudaPersistentDagTask *task;
    unsigned long long i;
};
""".strip()
COMBINE_WEIGHTS = (0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class TaskBodySpec:
    func_id: int
    task_name: str
    body: str
    source_kind: str = "persistent-task-body"
    source_sha256: str = ""

    def with_digest(self) -> "TaskBodySpec":
        if self.source_sha256:
            return self
        return TaskBodySpec(
            func_id=self.func_id,
            task_name=self.task_name,
            body=self.body,
            source_kind=self.source_kind,
            source_sha256=sha256(self.body.encode("utf-8")).hexdigest(),
        )


class PtoRunTiming(ctypes.Structure):
    _fields_ = [
        ("host_wall_ns", ctypes.c_uint64),
        ("device_wall_ns", ctypes.c_uint64),
    ]


def build_task_body_specs() -> list[TaskBodySpec]:
    moe_body = generate_gluon_persistent_task_body("moe_expert_affine_f32")
    return [
        TaskBodySpec(
            func_id=12,
            task_name=moe_body.task_name,
            body=moe_body.body,
            source_kind=moe_body.source_kind,
            source_sha256=moe_body.source_sha256,
        ),
        TaskBodySpec(
            func_id=4,
            task_name="axpy_f32",
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i] + task->b[i];
""".strip(),
        ).with_digest(),
        TaskBodySpec(
            func_id=11,
            task_name="scale_f32",
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i];
""".strip(),
        ).with_digest(),
        TaskBodySpec(
            func_id=2,
            task_name="mul_f32",
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->b[i];
""".strip(),
        ).with_digest(),
        TaskBodySpec(
            func_id=13,
            task_name="weighted_combine_f32",
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
if (task->scalar_arg_count < 4U) {
    return;
}
task->out[i] = task->scalar_args[0] * task->a[i] +
               task->scalar_args[1] * task->b[i] +
               task->scalar_args[2] * task->c[i] +
               task->scalar_args[3] * task->d[i];
""".strip(),
        ).with_digest(),
    ]


def gluon_expert_bridge_metadata(task_specs: list[TaskBodySpec]) -> dict:
    expert_spec = next(spec for spec in task_specs if spec.func_id == 12)
    return {
        "func_id": expert_spec.func_id,
        "kernel_name": "moe_expert_affine_f32",
        "task_name": expert_spec.task_name,
        "source_kind": expert_spec.source_kind,
        "source_sha256": expert_spec.source_sha256,
    }


def graph_descriptor(task_specs: list[TaskBodySpec] | None = None) -> dict:
    specs = build_task_body_specs() if task_specs is None else task_specs
    by_func_id = {spec.func_id: spec for spec in specs}
    tasks = [
        {
            "task_id": 0,
            "role": "expert_transform",
            "name": by_func_id[12].task_name,
            "func_id": 12,
            "depends_on": [],
            "dependents": [4],
            "initial_fanin": 0,
            "params": {"scale_a": 1.25, "scale_b": 0.5},
        },
        {
            "task_id": 1,
            "role": "expert_transform",
            "name": by_func_id[4].task_name,
            "func_id": 4,
            "depends_on": [],
            "dependents": [4],
            "initial_fanin": 0,
            "params": {"alpha": -0.75},
        },
        {
            "task_id": 2,
            "role": "expert_transform",
            "name": by_func_id[11].task_name,
            "func_id": 11,
            "depends_on": [],
            "dependents": [4],
            "initial_fanin": 0,
            "params": {"scale": 0.25},
        },
        {
            "task_id": 3,
            "role": "expert_transform",
            "name": by_func_id[2].task_name,
            "func_id": 2,
            "depends_on": [],
            "dependents": [4],
            "initial_fanin": 0,
            "params": {},
        },
        {
            "task_id": 4,
            "role": "weighted_combine",
            "name": by_func_id[13].task_name,
            "func_id": 13,
            "depends_on": [0, 1, 2, 3],
            "dependents": [],
            "initial_fanin": 4,
            "weights": list(COMBINE_WEIGHTS),
        },
    ]
    return {
        "dag_shape": DAG_SHAPE,
        "runtime": "persistent_device",
        "task_count": 5,
        "expert_task_count": 4,
        "combine_task_count": 1,
        "device_side_fanin_before_combine": True,
        "dependents": [4, 4, 4, 4],
        "fanin": [0, 0, 0, 0, 4],
        "tasks": tasks,
    }


def rendered_dispatch_source(task_specs: list[TaskBodySpec]) -> str:
    task_functions = [
        CudaPersistentTaskBodyFunction(
            func_id=spec.func_id,
            task_body=CudaTaskBody(
                name=spec.task_name,
                body=spec.body,
                context_definition=CONTEXT_DEFINITION,
            ),
        )
        for spec in task_specs
    ]
    return render_persistent_dag_source(task_functions)


def cpu_inputs(n: int) -> tuple[list[float], list[float]]:
    a = [float((idx % 17) - 8) * 0.125 for idx in range(n)]
    b = [float((idx % 19) - 9) * 0.0625 for idx in range(n)]
    return a, b


def cpu_golden(n: int) -> dict[str, list[float]]:
    a, b = cpu_inputs(n)
    expert0 = [1.25 * av + 0.5 * bv for av, bv in zip(a, b)]
    expert1 = [-0.75 * av + bv for av, bv in zip(a, b)]
    expert2 = [0.25 * av for av in a]
    expert3 = [av * bv for av, bv in zip(a, b)]
    out = [
        COMBINE_WEIGHTS[0] * e0
        + COMBINE_WEIGHTS[1] * e1
        + COMBINE_WEIGHTS[2] * e2
        + COMBINE_WEIGHTS[3] * e3
        for e0, e1, e2, e3 in zip(expert0, expert1, expert2, expert3)
    ]
    return {
        "a": a,
        "b": b,
        "expert0": expert0,
        "expert1": expert1,
        "expert2": expert2,
        "expert3": expert3,
        "out": out,
    }


def run_moe_dispatch_combine(
    *,
    device: int = 0,
    n: int = 4096,
    arch: str = "compute_80",
    block_dim: int = 256,
    scheduler_blocks: int = 1,
    worker_blocks: int = 4,
    queue_capacity: int = 5,
    stream_id: int = 0,
    skip_reason: Callable[[], str | None] | None = None,
) -> dict:
    if n <= 0:
        raise ValueError("--n must be positive")
    if queue_capacity < 5:
        raise ValueError("--queue-capacity must be at least 5")
    if scheduler_blocks <= 0 or worker_blocks <= 0:
        raise ValueError("--scheduler-blocks and --worker-blocks must be positive")

    task_specs = build_task_body_specs()
    dispatch_source = rendered_dispatch_source(task_specs)
    descriptor = graph_descriptor(task_specs)
    base = {
        "schema_version": 1,
        "status": "not_run",
        "runtime": "persistent_device",
        "dag_shape": DAG_SHAPE,
        "device": device,
        "n": n,
        "arch": arch,
        "block_dim": block_dim,
        "scheduler_blocks": scheduler_blocks,
        "worker_blocks": worker_blocks,
        "queue_capacity": queue_capacity,
        "stream_id": stream_id,
        "graph_descriptor": descriptor,
        "dispatch_source": {
            "source_kind": "generated-dispatch",
            "source_sha256": sha256(dispatch_source.encode("utf-8")).hexdigest(),
        },
        "gluon_expert_bridge": gluon_expert_bridge_metadata(task_specs),
        "task_bodies": [
            {
                "func_id": spec.func_id,
                "name": spec.task_name,
                "source_kind": spec.source_kind,
                "source_sha256": spec.source_sha256,
            }
            for spec in task_specs
        ],
        "non_claims": [
            "single-process synthetic persistent-device graph only",
            "no distributed expert parallelism or communication path",
            "no serving, vLLM, DeepSeek, or performance claim",
        ],
    }

    reason_check = cuda_skip_reason if skip_reason is None else skip_reason
    reason = reason_check()
    if reason is not None:
        return {
            **base,
            "status": "skipped",
            "reason": _clean_text(reason),
            "expected_preview": cpu_golden(min(n, 8))["out"],
        }

    return _run_cuda_graph(
        base=base,
        task_specs=task_specs,
        device=device,
        n=n,
        arch=arch,
        block_dim=block_dim,
        scheduler_blocks=scheduler_blocks,
        worker_blocks=worker_blocks,
        queue_capacity=queue_capacity,
        stream_id=stream_id,
    )


def run_two_device_moe_dispatch_combine(
    *,
    device_ids: tuple[int, int] = (6, 7),
    n: int = 4096,
    arch: str = "compute_80",
    block_dim: int = 256,
    scheduler_blocks: int = 1,
    worker_blocks: int = 4,
    queue_capacity: int = 5,
    stream_id: int = 0,
    runner: Callable[..., dict] | None = None,
) -> dict:
    if len(device_ids) != 2:
        raise ValueError("--device-ids must name exactly two CUDA devices")
    if len(set(device_ids)) != 2:
        raise ValueError("--device-ids must name two distinct CUDA devices")

    run_one = _run_single_device_subprocess if runner is None else runner
    per_device_results = []
    for device in device_ids:
        try:
            per_device_results.append(
                run_one(
                    device=device,
                    n=n,
                    arch=arch,
                    block_dim=block_dim,
                    scheduler_blocks=scheduler_blocks,
                    worker_blocks=worker_blocks,
                    queue_capacity=queue_capacity,
                    stream_id=stream_id,
                )
            )
        except Exception as exc:
            per_device_results.append(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "dag_shape": DAG_SHAPE,
                    "device": device,
                    "error_type": type(exc).__name__,
                    "error": _clean_text(str(exc)),
                }
            )

    validation = _validate_two_device_results(per_device_results)
    if all(validation.values()):
        status = "passed"
    elif all(result.get("status") == "skipped" for result in per_device_results):
        status = "skipped"
    else:
        status = "failed"

    return {
        "schema_version": 1,
        "status": status,
        "runtime": "persistent_device",
        "dag_shape": DAG_SHAPE,
        "evidence_scope": TWO_DEVICE_EVIDENCE_SCOPE,
        "evidence_statement": TWO_DEVICE_EVIDENCE_STATEMENT,
        "device_ids": list(device_ids),
        "devices": list(device_ids),
        "n": n,
        "arch": arch,
        "block_dim": block_dim,
        "scheduler_blocks": scheduler_blocks,
        "worker_blocks": worker_blocks,
        "queue_capacity": queue_capacity,
        "stream_id": stream_id,
        "per_device_count": len(per_device_results),
        "validation": validation,
        "source_digests": _two_device_source_digests(per_device_results),
        "per_device_results": per_device_results,
        "non_claims": [
            "same-node independent per-device persistent MoE graph runs only",
            "not fused cross-GPU expert-parallel MoE",
            "no distributed serving, vLLM, DeepSeek, RDMA, or performance claim",
        ],
    }


def run_persistent_moe_nccl_handoff(
    *,
    device_ids: tuple[int, int] = (6, 7),
    n: int = 4096,
    tensor_numel: int = 1024,
    arch: str = "compute_80",
    block_dim: int = 256,
    scheduler_blocks: int = 1,
    worker_blocks: int = 4,
    queue_capacity: int = 5,
    stream_id: int = 0,
    build: bool = False,
    moe_runner: Callable[..., dict] | None = None,
    nccl_runner: Callable[..., dict] | None = None,
) -> dict:
    if len(device_ids) != 2:
        raise ValueError("--device-ids must name exactly two CUDA devices")
    if len(set(device_ids)) != 2:
        raise ValueError("--device-ids must name two distinct CUDA devices")
    if tensor_numel <= 0:
        raise ValueError("--tensor-numel must be positive")

    run_moe = run_two_device_moe_dispatch_combine if moe_runner is None else moe_runner
    run_nccl = _run_nccl_worker_control_ops if nccl_runner is None else nccl_runner
    persistent_moe = run_moe(
        device_ids=device_ids,
        n=n,
        arch=arch,
        block_dim=block_dim,
        scheduler_blocks=scheduler_blocks,
        worker_blocks=worker_blocks,
        queue_capacity=queue_capacity,
        stream_id=stream_id,
    )
    nccl_worker_control = _review_safe_payload(
        run_nccl(device_ids=device_ids, tensor_numel=tensor_numel, build=build)
    )

    persistent_validation = persistent_moe.get("validation") or {}
    nccl_validation = _validate_nccl_worker_control_result(nccl_worker_control)
    source_digests = persistent_moe.get("source_digests") or {}
    handoff_validation = {
        "same_device_ids": (
            persistent_moe.get("device_ids") == list(device_ids)
            and nccl_worker_control.get("device_ids") == list(device_ids)
        ),
        "persistent_moe_passed": persistent_moe.get("status") == "passed",
        "nccl_worker_control_passed": nccl_worker_control.get("status") == "passed",
        "persistent_moe_validation_passed": bool(persistent_validation)
        and all(bool(value) for value in persistent_validation.values()),
        "nccl_worker_control_validation_passed": all(
            bool(value) for value in nccl_validation.values()
        ),
        "source_digests_present": all(
            source_digests.get(name)
            for name in (
                "dispatch_source_sha256",
                "gluon_expert_bridge_sha256",
                "task_body_func12_sha256",
            )
        ),
        "bridge_digests_match": (
            source_digests.get("gluon_expert_bridge_sha256")
            == source_digests.get("task_body_func12_sha256")
            and bool(source_digests.get("gluon_expert_bridge_sha256"))
        ),
    }

    if all(handoff_validation.values()):
        status = "passed"
    elif persistent_moe.get("status") == "skipped" or nccl_worker_control.get("status") == "skipped":
        status = "skipped"
    else:
        status = "failed"

    return {
        "schema_version": 1,
        "status": status,
        "handoff_scope": HANDOFF_SCOPE,
        "evidence_statement": HANDOFF_EVIDENCE_STATEMENT,
        "device_ids": list(device_ids),
        "devices": list(device_ids),
        "n": n,
        "tensor_numel": int(tensor_numel),
        "arch": arch,
        "block_dim": block_dim,
        "scheduler_blocks": scheduler_blocks,
        "worker_blocks": worker_blocks,
        "queue_capacity": queue_capacity,
        "stream_id": stream_id,
        "persistent_moe_validation": persistent_validation,
        "persistent_moe_source_digests": source_digests,
        "persistent_moe_max_abs_error": _max_result_error(persistent_moe.get("per_device_results") or []),
        "persistent_moe_scheduler_errors": _persistent_scheduler_errors(persistent_moe),
        "nccl_worker_control_validation": nccl_validation,
        "nccl_worker_control_max_abs_error": _max_nccl_error(nccl_worker_control),
        "handoff_validation": handoff_validation,
        "handoff_boundary": {
            "persistent_moe_scope": persistent_moe.get("evidence_scope"),
            "nccl_transport": nccl_worker_control.get("transport"),
            "nccl_capability_id": (nccl_worker_control.get("capability") or {}).get(
                "capability_id"
            ),
            "source_digests": source_digests,
            "same_device_ids": handoff_validation["same_device_ids"],
        },
        "persistent_moe": persistent_moe,
        "nccl_worker_control": nccl_worker_control,
        "non_claims": [
            "composition of existing persistent MoE and NCCL worker-control paths only",
            "not fused cross-GPU expert-parallel MoE",
            "no serving, vLLM, DeepSeek, RDMA, multi-node, or performance claim",
        ],
    }


def run_persistent_moe_uccl_ep_handoff(
    *,
    device_ids: tuple[int, int] = (6, 7),
    n: int = 4096,
    tensor_numel: int = 1024,
    arch: str = "compute_80",
    block_dim: int = 256,
    scheduler_blocks: int = 1,
    worker_blocks: int = 4,
    queue_capacity: int = 5,
    stream_id: int = 0,
    num_tokens: int = 64,
    num_topk: int = 4,
    num_experts: int = 16,
    input_dtype: str = "bf16",
    repeats: int = 1,
    uccl_ep_bench_dir: str | None = None,
    moe_runner: Callable[..., dict] | None = None,
    uccl_ep_runner: Callable[..., dict] | None = None,
) -> dict:
    if len(device_ids) != 2:
        raise ValueError("--device-ids must name exactly two CUDA devices")
    if len(set(device_ids)) != 2:
        raise ValueError("--device-ids must name two distinct CUDA devices")
    if tensor_numel <= 0:
        raise ValueError("--tensor-numel must be positive")

    run_moe = run_two_device_moe_dispatch_combine if moe_runner is None else moe_runner
    run_uccl_ep = _run_uccl_ep_dispatch_combine_adapter if uccl_ep_runner is None else uccl_ep_runner
    persistent_moe = run_moe(
        device_ids=device_ids,
        n=n,
        arch=arch,
        block_dim=block_dim,
        scheduler_blocks=scheduler_blocks,
        worker_blocks=worker_blocks,
        queue_capacity=queue_capacity,
        stream_id=stream_id,
    )
    uccl_ep_adapter = _review_safe_payload(
        run_uccl_ep(
            device_ids=device_ids,
            num_tokens=num_tokens,
            hidden=tensor_numel,
            num_topk=num_topk,
            num_experts=num_experts,
            input_dtype=input_dtype,
            repeats=repeats,
            bench_dir=uccl_ep_bench_dir,
        )
    )

    persistent_validation = persistent_moe.get("validation") or {}
    uccl_ep_validation = _validate_uccl_ep_adapter_result(uccl_ep_adapter)
    source_digests = persistent_moe.get("source_digests") or {}
    max_abs_error = _max_uccl_ep_error(uccl_ep_adapter, "max_abs_error")
    topk_weight_error = _max_uccl_ep_error(uccl_ep_adapter, "topk_weight_error")
    payload_provenance = _payload_provenance(
        persistent_moe=persistent_moe,
        uccl_ep_adapter=uccl_ep_adapter,
        device_ids=device_ids,
    )
    handoff_validation = {
        "same_device_ids": (
            persistent_moe.get("device_ids") == list(device_ids)
            and uccl_ep_adapter.get("device_ids") == list(device_ids)
        ),
        "persistent_moe_passed": persistent_moe.get("status") == "passed",
        "uccl_ep_adapter_passed": uccl_ep_adapter.get("status") == "passed",
        "persistent_moe_validation_passed": bool(persistent_validation)
        and all(bool(value) for value in persistent_validation.values()),
        "uccl_ep_adapter_validation_passed": all(
            bool(value) for value in uccl_ep_validation.values()
        ),
        "source_digests_present": all(
            source_digests.get(name)
            for name in (
                "dispatch_source_sha256",
                "gluon_expert_bridge_sha256",
                "task_body_func12_sha256",
            )
        ),
        "bridge_digests_match": (
            source_digests.get("gluon_expert_bridge_sha256")
            == source_digests.get("task_body_func12_sha256")
            and bool(source_digests.get("gluon_expert_bridge_sha256"))
        ),
        "adapter_descriptor_metadata_present": uccl_ep_validation[
            "descriptor_metadata_present"
        ],
        "max_errors_zero": max_abs_error == 0.0 and topk_weight_error == 0.0,
    }

    if all(handoff_validation.values()):
        status = "passed"
    elif persistent_moe.get("status") == "skipped" or uccl_ep_adapter.get("status") == "skipped":
        status = "skipped"
    else:
        status = "failed"

    return {
        "schema_version": 1,
        "status": status,
        "handoff_scope": UCCL_EP_HANDOFF_SCOPE,
        "evidence_statement": UCCL_EP_HANDOFF_EVIDENCE_STATEMENT,
        "device_ids": list(device_ids),
        "devices": list(device_ids),
        "n": n,
        "tensor_numel": int(tensor_numel),
        "arch": arch,
        "block_dim": block_dim,
        "scheduler_blocks": scheduler_blocks,
        "worker_blocks": worker_blocks,
        "queue_capacity": queue_capacity,
        "stream_id": stream_id,
        "uccl_ep_adapter_shape": {
            "num_tokens": int(num_tokens),
            "hidden": int(tensor_numel),
            "num_topk": int(num_topk),
            "num_experts": int(num_experts),
            "input_dtype": input_dtype,
            "repeats": int(repeats),
        },
        "persistent_moe_validation": persistent_validation,
        "persistent_moe_source_digests": source_digests,
        "persistent_moe_max_abs_error": _max_result_error(persistent_moe.get("per_device_results") or []),
        "persistent_moe_scheduler_errors": _persistent_scheduler_errors(persistent_moe),
        "uccl_ep_adapter_validation": uccl_ep_validation,
        "uccl_ep_adapter_max_abs_error": max_abs_error,
        "uccl_ep_adapter_topk_weight_error": topk_weight_error,
        "handoff_validation": handoff_validation,
        "payload_provenance": payload_provenance,
        "handoff_boundary": {
            "persistent_moe_scope": persistent_moe.get("evidence_scope"),
            "uccl_backend": uccl_ep_adapter.get("backend"),
            "uccl_transport": uccl_ep_adapter.get("transport"),
            "uccl_operation": uccl_ep_adapter.get("operation"),
            "uccl_capability_id": (uccl_ep_adapter.get("capability") or {}).get(
                "capability_id"
            ),
            "uccl_descriptor": uccl_ep_adapter.get("descriptor"),
            "source_digests": source_digests,
            "same_device_ids": handoff_validation["same_device_ids"],
        },
        "persistent_moe": persistent_moe,
        "uccl_ep_adapter": uccl_ep_adapter,
        "non_claims": [
            "composition of existing persistent MoE and UCCL-EP adapter paths only",
            "not fused cross-GPU expert-parallel MoE",
            "not CUDA host-runtime UCCL dispatch",
            "no serving, vLLM, DeepSeek, RDMA, multi-node, or performance claim",
        ],
    }


def run_persistent_moe_uccl_ep_fused_boundary(
    *,
    device_ids: tuple[int, int] = (6, 7),
    n: int = 4096,
    tensor_numel: int = 1024,
    arch: str = "compute_80",
    block_dim: int = 256,
    scheduler_blocks: int = 1,
    worker_blocks: int = 4,
    queue_capacity: int = 5,
    stream_id: int = 0,
    num_tokens: int = 64,
    num_topk: int = 4,
    num_experts: int = 16,
    input_dtype: str = "bf16",
    repeats: int = 1,
    uccl_ep_bench_dir: str | None = None,
    moe_runner: Callable[..., dict] | None = None,
    uccl_ep_runner: Callable[..., dict] | None = None,
) -> dict:
    handoff = run_persistent_moe_uccl_ep_handoff(
        device_ids=device_ids,
        n=n,
        tensor_numel=tensor_numel,
        arch=arch,
        block_dim=block_dim,
        scheduler_blocks=scheduler_blocks,
        worker_blocks=worker_blocks,
        queue_capacity=queue_capacity,
        stream_id=stream_id,
        num_tokens=num_tokens,
        num_topk=num_topk,
        num_experts=num_experts,
        input_dtype=input_dtype,
        repeats=repeats,
        uccl_ep_bench_dir=uccl_ep_bench_dir,
        moe_runner=moe_runner,
        uccl_ep_runner=uccl_ep_runner,
    )
    handoff_validation = handoff.get("handoff_validation") or {}
    expected_rank_to_device = _rank_to_device_map(device_ids)
    runtime_fusion = _runtime_fusion_status_from_handoff(
        handoff=handoff,
        expected_device_ids=device_ids,
        expected_rank_to_device=expected_rank_to_device,
        expected_descriptor=handoff.get("uccl_ep_adapter_shape") or {},
    )
    boundary_validation = {
        "handoff_passed": handoff.get("status") == "passed",
        "same_device_ids": bool(handoff_validation.get("same_device_ids")),
        "persistent_moe_passed": bool(handoff_validation.get("persistent_moe_passed")),
        "uccl_ep_adapter_passed": bool(
            handoff_validation.get("uccl_ep_adapter_passed")
        ),
        "descriptor_metadata_present": bool(
            handoff_validation.get("adapter_descriptor_metadata_present")
        ),
        "runtime_fusion_guard_passed": runtime_fusion["status"] == "passed",
        "actual_fused_cross_gpu_execution": bool(
            runtime_fusion["actual_fused_cross_gpu_execution"]
        ),
        "persistent_device_uccl_ep_runtime_fusion": runtime_fusion["status"] == "passed",
        "structured_unsupported_boundary": (
            handoff.get("status") == "passed" and runtime_fusion["status"] == "unsupported"
        ),
        "rejected_pass_evidence": runtime_fusion["status"] == "failed",
    }
    if runtime_fusion["status"] == "passed":
        status = "passed"
    elif runtime_fusion["status"] == "failed":
        status = "failed"
    elif handoff.get("status") == "passed":
        status = "unsupported"
    else:
        status = handoff.get("status", "failed")

    return {
        "schema_version": 1,
        "status": status,
        "runtime": "persistent_device",
        "dag_shape": DAG_SHAPE,
        "fused_boundary_scope": UCCL_EP_FUSED_BOUNDARY_SCOPE,
        "handoff_scope": handoff.get("handoff_scope"),
        "evidence_statement": UCCL_EP_FUSED_BOUNDARY_EVIDENCE_STATEMENT,
        "device_ids": list(device_ids),
        "devices": list(device_ids),
        "n": n,
        "tensor_numel": int(tensor_numel),
        "arch": arch,
        "block_dim": block_dim,
        "scheduler_blocks": scheduler_blocks,
        "worker_blocks": worker_blocks,
        "queue_capacity": queue_capacity,
        "stream_id": stream_id,
        "uccl_ep_adapter_shape": handoff.get("uccl_ep_adapter_shape"),
        "boundary_validation": boundary_validation,
        "persistent_device_uccl_ep_runtime_fusion": runtime_fusion,
        "missing_boundaries": list(UCCL_EP_FUSED_BOUNDARY_MISSING),
        "payload_provenance": handoff.get("payload_provenance"),
        "uccl_ep_handoff": handoff,
        "handoff_boundary": handoff.get("handoff_boundary"),
        "persistent_moe_validation": handoff.get("persistent_moe_validation"),
        "uccl_ep_adapter_validation": handoff.get("uccl_ep_adapter_validation"),
        "persistent_moe_source_digests": handoff.get("persistent_moe_source_digests"),
        "non_claims": [
            "structured unsupported boundary, not fused MoE evidence",
            "not actual fused cross-GPU expert-parallel MoE execution",
            "not CUDA host-runtime UCCL dispatch",
            "no serving, vLLM, DeepSeek, RDMA, multi-node, throughput, or latency claim",
        ],
    }


def _run_nccl_worker_control_ops(**kwargs) -> dict:
    module_path = Path(__file__).with_name("nccl_worker_control_ops.py")
    spec = importlib.util.spec_from_file_location("nccl_worker_control_ops_example", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run_worker_control_ops(**kwargs)


def _run_uccl_ep_dispatch_combine_adapter(**kwargs) -> dict:
    module_path = Path(__file__).with_name("uccl_ep_dispatch_combine_adapter.py")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(module_path),
        "--device-ids",
        ",".join(str(device) for device in kwargs["device_ids"]),
        "--num-tokens",
        str(kwargs["num_tokens"]),
        "--hidden",
        str(kwargs["hidden"]),
        "--num-topk",
        str(kwargs["num_topk"]),
        "--num-experts",
        str(kwargs["num_experts"]),
        "--input-dtype",
        str(kwargs["input_dtype"]),
        "--repeats",
        str(kwargs["repeats"]),
        "--require-cuda",
    ]
    if kwargs.get("bench_dir"):
        command.extend(["--uccl-ep-bench-dir", str(kwargs["bench_dir"])])
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = _last_json_object(completed.stdout)
    if payload is None:
        return {
            "status": "failed",
            "backend": "uccl",
            "transport": "ep",
            "operation": "ep_dispatch_combine",
            "device_ids": list(kwargs["device_ids"]),
            "error_type": "JSONDecodeError",
            "error": "UCCL-EP adapter child did not emit JSON",
            "child_returncode": completed.returncode,
            "child_stderr": _clean_text(completed.stderr.strip()),
        }
    return {
        **payload,
        "child_returncode": completed.returncode,
        "child_stderr": _clean_text(completed.stderr.strip()),
    }


def _validate_nccl_worker_control_result(result: dict) -> dict[str, bool]:
    return {
        "all_reduce_passed": bool((result.get("all_reduce") or {}).get("passed")),
        "reduce_scatter_passed": bool((result.get("reduce_scatter") or {}).get("passed")),
        "all_gather_passed": bool((result.get("all_gather") or {}).get("passed")),
        "send_recv_passed": bool((result.get("send_recv") or {}).get("passed")),
        "max_abs_error_zero": _max_nccl_error(result) == 0.0,
    }


def _validate_uccl_ep_adapter_result(result: dict) -> dict[str, bool]:
    descriptor = result.get("descriptor") or {}
    rank_results = result.get("rank_results") or []
    return {
        "adapter_passed": result.get("status") == "passed",
        "transport_is_ep": result.get("transport") == "ep",
        "operation_is_dispatch_combine": result.get("operation") == "ep_dispatch_combine",
        "descriptor_metadata_present": all(
            descriptor.get(name)
            for name in ("num_tokens", "hidden", "num_topk", "num_experts", "input_dtype")
        ),
        "all_ranks_passed": bool(rank_results)
        and all(bool(item.get("passed")) for item in rank_results),
        "max_abs_error_zero": _max_uccl_ep_error(result, "max_abs_error") == 0.0,
        "topk_weight_error_zero": _max_uccl_ep_error(result, "topk_weight_error") == 0.0,
    }


def _max_nccl_error(result: dict) -> float:
    errors = [
        float((result.get(name) or {}).get("max_abs_error", float("inf")))
        for name in ("all_reduce", "reduce_scatter", "all_gather", "send_recv")
    ]
    return max(errors, default=float("inf"))


def _max_uccl_ep_error(result: dict, key: str) -> float:
    errors = [float(item.get(key, float("inf"))) for item in result.get("rank_results") or []]
    return max(errors, default=float("inf"))


def _max_result_error(results: list[dict]) -> float:
    errors = [float(result.get("max_abs_error", float("inf"))) for result in results]
    return max(errors, default=float("inf"))


def _persistent_scheduler_errors(result: dict) -> list[dict]:
    errors = []
    for item in result.get("per_device_results") or []:
        errors.append(
            {
                "device": item.get("device"),
                "errors": item.get("device_scheduler_errors"),
            }
        )
    return errors


def _payload_provenance(
    *,
    persistent_moe: dict,
    uccl_ep_adapter: dict,
    device_ids: tuple[int, ...],
) -> dict:
    source_digests = persistent_moe.get("source_digests") or {}
    return {
        "uccl_ep_adapter": _uccl_ep_adapter_provenance(uccl_ep_adapter, device_ids),
        "persistent_device_graph": {
            "producer": "persistent_moe_dispatch_combine",
            "graph_descriptor_id": persistent_moe.get("dag_shape") or DAG_SHAPE,
            "runtime": persistent_moe.get("runtime") or "persistent_device",
            "device_ids": list(persistent_moe.get("device_ids") or device_ids),
            "rank_to_device": _rank_to_device_map(device_ids),
            "source_digests": source_digests,
            "bridge_digest": source_digests.get("gluon_expert_bridge_sha256"),
        },
        "shared_payload_ownership": _absent_shared_payload_ownership(),
    }


def _uccl_ep_adapter_provenance(uccl_ep_adapter: dict, device_ids: tuple[int, ...]) -> dict:
    capability = uccl_ep_adapter.get("capability") or {}
    descriptor = uccl_ep_adapter.get("descriptor") or {}
    rank_results = uccl_ep_adapter.get("rank_results") or []
    return {
        "producer": "uccl_ep_dispatch_combine_adapter",
        "capability_id": capability.get("capability_id"),
        "world_size": uccl_ep_adapter.get("world_size") or capability.get("world_size"),
        "device_ids": list(uccl_ep_adapter.get("device_ids") or device_ids),
        "rank_to_device": capability.get("rank_to_device") or _rank_to_device_map(device_ids),
        "descriptor": {
            key: descriptor.get(key)
            for key in (
                "num_tokens",
                "hidden",
                "num_topk",
                "num_experts",
                "experts_per_rank",
                "input_dtype",
                "metadata_shapes",
            )
            if descriptor.get(key) is not None
        },
        "rank_results": [
            {
                key: item.get(key)
                for key in (
                    "rank",
                    "device_id",
                    "input_dtype",
                    "recv_tokens",
                    "expected_total_sent_tokens",
                    "passed",
                    "max_abs_error",
                    "topk_weight_error",
                )
                if item.get(key) is not None
            }
            for item in rank_results
        ],
    }


def _rank_to_device_map(device_ids: tuple[int, ...]) -> dict[str, int]:
    return {str(rank): int(device_id) for rank, device_id in enumerate(device_ids)}


def _absent_shared_payload_ownership() -> dict:
    return {
        "exists": False,
        "ownership_token": None,
        "lifetime_transition_log": [],
        "reason": NO_SHARED_PAYLOAD_OWNERSHIP_REASON,
    }


def _unsupported_runtime_fusion_status() -> dict:
    return {
        "status": "unsupported",
        "actual_fused_cross_gpu_execution": False,
        "shared_ownership_token": None,
        "payload_lifetime_transition_log": [],
        "reason": NO_SHARED_PAYLOAD_OWNERSHIP_REASON,
        "failure_fields": {
            "unsupported_boundary": "persistent_device_uccl_ep_runtime_fusion",
        },
    }


def _runtime_fusion_status_from_handoff(
    *,
    handoff: dict,
    expected_device_ids: tuple[int, ...],
    expected_rank_to_device: dict[str, int],
    expected_descriptor: dict,
) -> dict:
    candidate = _runtime_fusion_candidate_from_handoff(handoff)
    if candidate is None:
        return _unsupported_runtime_fusion_status()
    return _validate_runtime_fusion_evidence(
        candidate,
        expected_device_ids=expected_device_ids,
        expected_rank_to_device=expected_rank_to_device,
        expected_descriptor=expected_descriptor,
        trusted_runtime_source=False,
    )


def _runtime_fusion_candidate_from_handoff(handoff: dict) -> dict | None:
    candidates = [
        handoff.get("persistent_device_uccl_ep_runtime_fusion"),
        (handoff.get("uccl_ep_adapter") or {}).get("persistent_device_uccl_ep_runtime_fusion"),
    ]
    ownership = ((handoff.get("payload_provenance") or {}).get("shared_payload_ownership") or {})
    if ownership.get("exists") or ownership.get("ownership_token") or ownership.get("lifetime_transition_log"):
        candidates.append(
            {
                "producer": "payload_provenance",
                "status": "passed" if ownership.get("exists") else "unsupported",
                "shared_ownership_token": ownership.get("ownership_token"),
                "payload_lifetime_transition_log": ownership.get("lifetime_transition_log") or [],
            }
        )
    return next((candidate for candidate in candidates if isinstance(candidate, dict)), None)


def _validate_runtime_fusion_evidence(
    evidence: dict | None,
    *,
    expected_device_ids: tuple[int, ...],
    expected_rank_to_device: dict[str, int],
    expected_descriptor: dict,
    trusted_runtime_source: bool = True,
) -> dict:
    if not evidence:
        return _unsupported_runtime_fusion_status()

    failure_fields: dict[str, str] = {}
    token = evidence.get("shared_ownership_token")
    transition_log = evidence.get("payload_lifetime_transition_log") or []
    dispatch_descriptor = evidence.get("dispatch_payload_descriptor") or {}
    combine_descriptor = evidence.get("combine_payload_descriptor") or {}

    if (
        not trusted_runtime_source
        or evidence.get("producer") != RUNTIME_FUSION_PRODUCER
        or evidence.get("runtime_owned_descriptor") is not True
    ):
        failure_fields["fabricated_or_untrusted_pass_evidence"] = (
            "runtime fusion pass evidence must be emitted by the runtime-owned "
            "persistent-device/UCCL-EP fusion coordinator"
        )
    if not token:
        failure_fields["missing_ownership_token"] = "runtime fusion evidence is missing a shared token"
    if evidence.get("rank_to_device") != expected_rank_to_device:
        failure_fields["rank_device_mismatch"] = "runtime fusion rank/device mapping does not match"
    if list(evidence.get("device_ids") or []) != list(expected_device_ids):
        failure_fields["rank_device_mismatch"] = "runtime fusion device ids do not match"
    if int(evidence.get("world_size") or 0) != len(expected_device_ids):
        failure_fields["rank_device_mismatch"] = "runtime fusion world size does not match"

    _validate_runtime_fusion_descriptors(
        failure_fields=failure_fields,
        token=token,
        dispatch_descriptor=dispatch_descriptor,
        combine_descriptor=combine_descriptor,
        expected_descriptor=expected_descriptor,
    )
    _validate_runtime_fusion_lifetime_log(
        failure_fields=failure_fields,
        token=token,
        transition_log=transition_log,
    )
    if evidence.get("status") != "passed" or evidence.get("actual_fused_cross_gpu_execution") is not True:
        failure_fields["incomplete_pass_evidence"] = (
            "runtime fusion evidence must explicitly report passed and actual fused execution"
        )

    if failure_fields:
        return {
            "status": "failed",
            "actual_fused_cross_gpu_execution": False,
            "shared_ownership_token": token,
            "payload_lifetime_transition_log": transition_log,
            "reason": "runtime fusion evidence failed local guard validation",
            "failure_fields": failure_fields,
        }

    return {
        "status": "passed",
        "actual_fused_cross_gpu_execution": True,
        "shared_ownership_token": token,
        "payload_lifetime_transition_log": transition_log,
        "rank_to_device": dict(expected_rank_to_device),
        "device_ids": list(expected_device_ids),
        "world_size": len(expected_device_ids),
        "dispatch_payload_descriptor": dispatch_descriptor,
        "combine_payload_descriptor": combine_descriptor,
        "failure_fields": {},
    }


def _validate_runtime_fusion_descriptors(
    *,
    failure_fields: dict[str, str],
    token: str | None,
    dispatch_descriptor: dict,
    combine_descriptor: dict,
    expected_descriptor: dict,
) -> None:
    if not dispatch_descriptor or not combine_descriptor:
        failure_fields["descriptor_mismatch"] = "dispatch and combine payload descriptors are required"
        return
    for key in ("num_tokens", "hidden", "num_topk", "num_experts", "input_dtype"):
        expected = expected_descriptor.get(key)
        if expected is None:
            continue
        if dispatch_descriptor.get(key) != expected or combine_descriptor.get(key) != expected:
            failure_fields["descriptor_mismatch"] = f"payload descriptor field {key!r} does not match"
    for descriptor_name, descriptor in (
        ("dispatch_payload_descriptor", dispatch_descriptor),
        ("combine_payload_descriptor", combine_descriptor),
    ):
        if descriptor.get("ownership_token") != token:
            failure_fields["mismatched_ownership_token"] = (
                f"{descriptor_name} does not carry the shared ownership token"
            )


def _validate_runtime_fusion_lifetime_log(
    *,
    failure_fields: dict[str, str],
    token: str | None,
    transition_log: list[dict],
) -> None:
    if not transition_log:
        failure_fields["missing_lifetime_transition_log"] = "payload lifetime transition log is empty"
        return

    released_indices = [
        idx
        for idx, entry in enumerate(transition_log)
        if entry.get("state") == "released" or entry.get("owner") == "released"
    ]
    if len(released_indices) > 1:
        failure_fields["double_release"] = "payload ownership was released more than once"
    if released_indices and released_indices[0] != len(transition_log) - 1:
        failure_fields["use_after_release"] = "payload lifetime log continues after release"
    if not released_indices:
        failure_fields["leaked_in_flight_ownership"] = "payload ownership was never released"

    observed = tuple((entry.get("state"), entry.get("owner")) for entry in transition_log)
    if observed != RUNTIME_FUSION_LIFETIME_SEQUENCE:
        failure_fields.setdefault(
            "illegal_lifetime_transition",
            "payload lifetime transition log does not match the required sequence",
        )

    if token:
        for entry in transition_log:
            if entry.get("ownership_token") != token:
                failure_fields["mismatched_ownership_token"] = (
                    "payload lifetime transition uses a different ownership token"
                )
                break


def _validate_two_device_results(results: list[dict]) -> dict[str, bool]:
    return {
        "all_devices_passed": all(result.get("status") == "passed" for result in results),
        "completed_count_is_5": all(result.get("completed_count") == 5 for result in results),
        "scheduler_errors_zero": all(
            result.get("device_scheduler_errors") == {"count": 0, "code": 0, "task_id": 0}
            for result in results
        ),
        "fanin_remaining_zero": all(
            result.get("fanin_remaining") == [0, 0, 0, 0, 0] for result in results
        ),
        "source_digests_match": _matching_source_digests(results),
        "bridge_metadata_match": _matching_bridge_metadata(results),
    }


def _matching_source_digests(results: list[dict]) -> bool:
    digests = [_dispatch_source_sha256(result) for result in results]
    return all(digests) and len(set(digests)) == 1


def _matching_bridge_metadata(results: list[dict]) -> bool:
    bridges = [result.get("gluon_expert_bridge") for result in results]
    task_body_digests = [_task_body_sha256(result, 12) for result in results]
    return (
        all(isinstance(bridge, dict) for bridge in bridges)
        and len({json.dumps(bridge, sort_keys=True) for bridge in bridges}) == 1
        and all(task_body_digests)
        and len(set(task_body_digests)) == 1
        and bridges[0].get("source_sha256") == task_body_digests[0]
    )


def _two_device_source_digests(results: list[dict]) -> dict[str, str | None]:
    first = results[0] if results else {}
    bridge = first.get("gluon_expert_bridge") or {}
    return {
        "dispatch_source_sha256": _dispatch_source_sha256(first),
        "gluon_expert_bridge_sha256": bridge.get("source_sha256"),
        "task_body_func12_sha256": _task_body_sha256(first, 12),
    }


def _dispatch_source_sha256(result: dict) -> str | None:
    dispatch_source = result.get("dispatch_source") or {}
    return dispatch_source.get("source_sha256")


def _task_body_sha256(result: dict, func_id: int) -> str | None:
    for item in result.get("task_bodies") or []:
        if item.get("func_id") == func_id:
            return item.get("source_sha256")
    return None


def _run_single_device_subprocess(**kwargs) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--device",
        str(kwargs["device"]),
        "--n",
        str(kwargs["n"]),
        "--arch",
        str(kwargs["arch"]),
        "--block-dim",
        str(kwargs["block_dim"]),
        "--scheduler-blocks",
        str(kwargs["scheduler_blocks"]),
        "--worker-blocks",
        str(kwargs["worker_blocks"]),
        "--queue-capacity",
        str(kwargs["queue_capacity"]),
        "--stream-id",
        str(kwargs["stream_id"]),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "schema_version": 1,
            "status": "failed",
            "dag_shape": DAG_SHAPE,
            "device": kwargs["device"],
            "error_type": "JSONDecodeError",
            "error": "single-device child did not emit JSON",
            "child_returncode": completed.returncode,
            "child_stderr": _clean_text(completed.stderr.strip()),
        }
    return {
        **payload,
        "device": kwargs["device"],
        "child_returncode": completed.returncode,
        "child_stderr": _clean_text(completed.stderr.strip()),
    }


def _run_cuda_graph(
    *,
    base: dict,
    task_specs: list[TaskBodySpec],
    device: int,
    n: int,
    arch: str,
    block_dim: int,
    scheduler_blocks: int,
    worker_blocks: int,
    queue_capacity: int,
    stream_id: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="pto-moe-dispatch-combine-") as tmp:
        work_dir = Path(tmp)
        task_sources = []
        for spec in task_specs:
            source_path = work_dir / f"{spec.func_id}_{spec.task_name}.pto.cu"
            source_path.write_text(spec.body, encoding="utf-8")
            task_sources.append(
                {
                    "func_id": spec.func_id,
                    "task_name": spec.task_name,
                    "source_path": str(source_path),
                    "body_style": "task_body",
                    "context_definition": CONTEXT_DEFINITION,
                }
            )
        artifact = KernelCompiler(platform="cuda").compile_cuda_persistent_device(
            task_sources,
            arch=arch,
        )

    binaries = RuntimeBuilder(platform="cuda").get_binaries("persistent_device", build=True)
    runtime = ctypes.CDLL(str(binaries.host_path))
    _bind_runtime(runtime)

    prepared = prepare_cuda_persistent_device_callable(
        artifact,
        grid_dim=scheduler_blocks + worker_blocks,
        block_dim=block_dim,
        stream_id=stream_id,
    )
    ctx = runtime.create_device_context()
    if not ctx:
        raise RuntimeError("create_device_context returned null")

    allocations: list[int] = []
    registered = False
    try:
        if runtime.simpler_init(ctx, device, None, 0, None, 0) != 0:
            raise RuntimeError(f"simpler_init failed for CUDA device {device}")

        data = cpu_golden(n)
        float_array_t = ctypes.c_float * n
        host_a = float_array_t(*data["a"])
        host_b = float_array_t(*data["b"])
        host_out = float_array_t()
        nbytes = ctypes.sizeof(host_a)

        def malloc(size: int) -> int:
            ptr = runtime.device_malloc_ctx(ctx, size)
            if not ptr:
                raise RuntimeError(f"device_malloc_ctx failed for {size} bytes")
            allocations.append(ptr)
            return ptr

        def copy_to_device(dev_ptr: int, host_obj) -> None:
            if runtime.copy_to_device_ctx(
                ctx, dev_ptr, ctypes.byref(host_obj), ctypes.sizeof(host_obj)
            ) != 0:
                raise RuntimeError("copy_to_device_ctx failed")

        dev_a = malloc(nbytes)
        dev_b = malloc(nbytes)
        dev_tmp0 = malloc(nbytes)
        dev_tmp1 = malloc(nbytes)
        dev_tmp2 = malloc(nbytes)
        dev_tmp3 = malloc(nbytes)
        dev_out = malloc(nbytes)
        copy_to_device(dev_a, host_a)
        copy_to_device(dev_b, host_b)

        state, state_allocations = _build_device_state(
            runtime=runtime,
            ctx=ctx,
            malloc=malloc,
            n=n,
            queue_capacity=queue_capacity,
            scheduler_blocks=scheduler_blocks,
            dev_a=dev_a,
            dev_b=dev_b,
            dev_tmp0=dev_tmp0,
            dev_tmp1=dev_tmp1,
            dev_tmp2=dev_tmp2,
            dev_tmp3=dev_tmp3,
            dev_out=dev_out,
        )
        dev_state = malloc(ctypes.sizeof(state))
        copy_to_device(dev_state, state)

        args = CudaPersistentDagArgs(state=dev_state)
        timing = PtoRunTiming()
        if runtime.prepare_callable(ctx, 0, prepared.byref()) != 0:
            raise RuntimeError("prepare_callable failed for persistent MoE graph")
        registered = True
        if (
            runtime.run_prepared(
                ctx,
                None,
                0,
                ctypes.byref(args),
                block_dim,
                0,
                0,
                0,
                0,
                0,
                None,
                ctypes.byref(timing),
            )
            != 0
        ):
            raise RuntimeError("run_prepared failed for persistent MoE graph")
        if runtime.copy_from_device_ctx(ctx, ctypes.byref(host_out), dev_out, nbytes) != 0:
            raise RuntimeError("copy_from_device_ctx failed for output")

        actual = list(host_out)
        expected = data["out"]
        max_abs_error = max(abs(av - ev) for av, ev in zip(actual, expected))
        completed_count = _copy_u32_from_device(runtime, ctx, state_allocations["completed_count"])
        error_count = _copy_u32_from_device(runtime, ctx, state_allocations["error_count"])
        error_code = _copy_u32_from_device(runtime, ctx, state_allocations["error_code"])
        error_task_id = _copy_u32_from_device(runtime, ctx, state_allocations["error_task_id"])
        fanin_remaining = _copy_u32_array_from_device(runtime, ctx, state_allocations["fanin"], 5)

        passed = (
            max_abs_error <= 1e-5
            and completed_count == 5
            and error_count == 0
            and fanin_remaining == [0, 0, 0, 0, 0]
        )
        return {
            **base,
            "status": "passed" if passed else "failed",
            "artifact": {
                "entry_name": artifact.entry_name,
                "source_kind": artifact.source_kind,
                "source_path": _relative_path(artifact.source_path),
                "manifest_path": _relative_path(artifact.manifest_path),
                "cache_hit": artifact.cache_hit,
            },
            "timing": {
                "host_wall_ns": int(timing.host_wall_ns),
                "device_wall_ns": int(timing.device_wall_ns),
            },
            "max_abs_error": max_abs_error,
            "completed_count": completed_count,
            "device_scheduler_errors": {
                "count": error_count,
                "code": error_code,
                "task_id": error_task_id,
            },
            "fanin_remaining": fanin_remaining,
            "actual_preview": actual[:8],
            "expected_preview": expected[:8],
        }
    finally:
        if registered:
            runtime.unregister_callable(ctx, 0)
        for ptr in reversed(allocations):
            runtime.device_free_ctx(ctx, ptr)
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)


def _build_device_state(
    *,
    runtime,
    ctx: int,
    malloc: Callable[[int], int],
    n: int,
    queue_capacity: int,
    scheduler_blocks: int,
    dev_a: int,
    dev_b: int,
    dev_tmp0: int,
    dev_tmp1: int,
    dev_tmp2: int,
    dev_tmp3: int,
    dev_out: int,
) -> tuple[CudaPersistentDagState, dict[str, int]]:
    tensor_args_t = ctypes.c_void_p * 4
    scalar_args_t = ctypes.c_float * 4
    task_t = CudaPersistentDagTask * 5
    tasks = task_t(
        CudaPersistentDagTask(
            func_id=12,
            a=dev_a,
            b=dev_b,
            out=dev_tmp0,
            n=n,
            dependent_begin=0,
            dependent_count=1,
            initial_fanin=0,
            scalar0=1.25,
            scalar1=0.5,
        ),
        CudaPersistentDagTask(
            func_id=4,
            a=dev_a,
            b=dev_b,
            out=dev_tmp1,
            n=n,
            dependent_begin=1,
            dependent_count=1,
            initial_fanin=0,
            scalar0=-0.75,
        ),
        CudaPersistentDagTask(
            func_id=11,
            a=dev_a,
            out=dev_tmp2,
            n=n,
            dependent_begin=2,
            dependent_count=1,
            initial_fanin=0,
            scalar0=0.25,
        ),
        CudaPersistentDagTask(
            func_id=2,
            a=dev_a,
            b=dev_b,
            out=dev_tmp3,
            n=n,
            dependent_begin=3,
            dependent_count=1,
            initial_fanin=0,
        ),
        CudaPersistentDagTask(
            func_id=13,
            a=dev_tmp0,
            b=dev_tmp1,
            c=dev_tmp2,
            d=dev_tmp3,
            out=dev_out,
            n=n,
            dependent_begin=4,
            dependent_count=0,
            initial_fanin=4,
            tensor_args=tensor_args_t(dev_tmp0, dev_tmp1, dev_tmp2, dev_tmp3),
            scalar_args=scalar_args_t(*COMBINE_WEIGHTS),
            tensor_arg_count=4,
            scalar_arg_count=4,
        ),
    )
    fanin_t = ctypes.c_uint32 * 5
    dependents_t = ctypes.c_uint32 * 4
    queue_t = ctypes.c_uint32 * queue_capacity
    scheduler_t = ctypes.c_uint32 * scheduler_blocks
    host_buffers = {
        "tasks": tasks,
        "dependents": dependents_t(4, 4, 4, 4),
        "fanin": fanin_t(0, 0, 0, 0, 4),
        "ready_queue": queue_t(),
        "ready_flags": queue_t(),
        "completion_queue": queue_t(),
        "completion_flags": queue_t(),
        "queue_head": ctypes.c_uint32(0),
        "queue_tail": ctypes.c_uint32(0),
        "completion_head": ctypes.c_uint32(0),
        "completion_tail": ctypes.c_uint32(0),
        "completed_count": ctypes.c_uint32(0),
        "error_count": ctypes.c_uint32(0),
        "error_code": ctypes.c_uint32(0),
        "error_task_id": ctypes.c_uint32(0),
        "scheduler_init_count": ctypes.c_uint32(0),
        "scheduler_loop_count": ctypes.c_uint32(0),
        "scheduler_processed_count": ctypes.c_uint32(0),
        "scheduler_processed_by_block": scheduler_t(),
    }

    dev: dict[str, int] = {}
    for name, host_obj in host_buffers.items():
        ptr = malloc(ctypes.sizeof(host_obj))
        if runtime.copy_to_device_ctx(ctx, ptr, ctypes.byref(host_obj), ctypes.sizeof(host_obj)) != 0:
            raise RuntimeError(f"copy_to_device_ctx failed for {name}")
        dev[name] = ptr

    state = CudaPersistentDagState(
        tasks=dev["tasks"],
        task_count=5,
        dependents=dev["dependents"],
        dependent_count=4,
        fanin=dev["fanin"],
        ready_queue=dev["ready_queue"],
        ready_flags=dev["ready_flags"],
        completion_queue=dev["completion_queue"],
        completion_flags=dev["completion_flags"],
        queue_capacity=queue_capacity,
        queue_head=dev["queue_head"],
        queue_tail=dev["queue_tail"],
        completion_head=dev["completion_head"],
        completion_tail=dev["completion_tail"],
        completed_count=dev["completed_count"],
        error_count=dev["error_count"],
        error_code=dev["error_code"],
        error_task_id=dev["error_task_id"],
        scheduler_blocks=scheduler_blocks,
        scheduler_init_count=dev["scheduler_init_count"],
        scheduler_loop_count=dev["scheduler_loop_count"],
        scheduler_processed_count=dev["scheduler_processed_count"],
        scheduler_processed_by_block=dev["scheduler_processed_by_block"],
    )
    return state, dev


def _bind_runtime(runtime) -> None:
    runtime.create_device_context.restype = ctypes.c_void_p
    runtime.destroy_device_context.argtypes = [ctypes.c_void_p]
    runtime.simpler_init.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    runtime.simpler_init.restype = ctypes.c_int
    runtime.finalize_device.argtypes = [ctypes.c_void_p]
    runtime.finalize_device.restype = ctypes.c_int
    runtime.device_malloc_ctx.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    runtime.device_malloc_ctx.restype = ctypes.c_void_p
    runtime.device_free_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.copy_to_device_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    runtime.copy_to_device_ctx.restype = ctypes.c_int
    runtime.copy_from_device_ctx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    runtime.copy_from_device_ctx.restype = ctypes.c_int
    runtime.prepare_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    runtime.prepare_callable.restype = ctypes.c_int
    runtime.run_prepared.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(PtoRunTiming),
    ]
    runtime.run_prepared.restype = ctypes.c_int
    runtime.unregister_callable.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    runtime.unregister_callable.restype = ctypes.c_int


def _copy_u32_from_device(runtime, ctx: int, dev_ptr: int) -> int:
    value = ctypes.c_uint32()
    if runtime.copy_from_device_ctx(ctx, ctypes.byref(value), dev_ptr, ctypes.sizeof(value)) != 0:
        raise RuntimeError("copy_from_device_ctx failed for uint32")
    return int(value.value)


def _copy_u32_array_from_device(runtime, ctx: int, dev_ptr: int, count: int) -> list[int]:
    array_t = ctypes.c_uint32 * count
    values = array_t()
    if runtime.copy_from_device_ctx(ctx, ctypes.byref(values), dev_ptr, ctypes.sizeof(values)) != 0:
        raise RuntimeError("copy_from_device_ctx failed for uint32 array")
    return [int(value) for value in values]


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def parse_device_ids(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected exactly two comma-separated device ids")
    try:
        devices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("device ids must be integers") from exc
    if len(set(devices)) != 2:
        raise argparse.ArgumentTypeError("device ids must be distinct")
    return devices


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    output_json = DEFAULT_OUTPUT_JSON
    require_cuda = False
    try:
        parser = JsonArgumentParser(description=__doc__)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument(
            "--device-ids",
            type=parse_device_ids,
            help="run the same persistent MoE graph independently on two CUDA devices",
        )
        parser.add_argument("--n", type=int, default=4096)
        parser.add_argument("--arch", default="compute_80")
        parser.add_argument("--block-dim", type=int, default=256)
        parser.add_argument("--scheduler-blocks", type=int, default=1)
        parser.add_argument("--worker-blocks", type=int, default=4)
        parser.add_argument("--queue-capacity", type=int, default=5)
        parser.add_argument("--stream-id", type=int, default=0)
        parser.add_argument("--tensor-numel", type=int, default=1024)
        parser.add_argument(
            "--build",
            action="store_true",
            help="rebuild CUDA runtime before the NCCL handoff path runs",
        )
        parser.add_argument(
            "--with-nccl-handoff",
            action="store_true",
            help="run the two-device MoE baseline and NCCL worker-control path on the same devices",
        )
        parser.add_argument(
            "--with-uccl-ep-handoff",
            action="store_true",
            help="run the two-device MoE baseline and UCCL-EP adapter on the same devices",
        )
        parser.add_argument(
            "--with-uccl-ep-fused-boundary",
            action="store_true",
            help=(
                "run the UCCL-EP handoff path and report the reduced fused "
                "cross-GPU MoE boundary status"
            ),
        )
        parser.add_argument("--uccl-ep-num-tokens", type=int, default=64)
        parser.add_argument("--uccl-ep-num-topk", type=int, default=4)
        parser.add_argument("--uccl-ep-num-experts", type=int, default=16)
        parser.add_argument("--uccl-ep-input-dtype", choices=("bf16", "fp8"), default="bf16")
        parser.add_argument("--uccl-ep-repeats", type=int, default=1)
        parser.add_argument("--uccl-ep-bench-dir")
        parser.add_argument("--output-json", type=Path)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return non-zero when CUDA tooling or a CUDA device is unavailable",
        )
        args = parser.parse_args(raw_args)
        output_json = args.output_json
        require_cuda = args.require_cuda
        if args.with_nccl_handoff:
            if args.device_ids is None:
                raise ValueError("--with-nccl-handoff requires --device-ids")
            result = run_persistent_moe_nccl_handoff(
                device_ids=args.device_ids,
                n=args.n,
                tensor_numel=args.tensor_numel,
                arch=args.arch,
                block_dim=args.block_dim,
                scheduler_blocks=args.scheduler_blocks,
                worker_blocks=args.worker_blocks,
                queue_capacity=args.queue_capacity,
                stream_id=args.stream_id,
                build=args.build,
            )
        elif args.with_uccl_ep_handoff:
            if args.device_ids is None:
                raise ValueError("--with-uccl-ep-handoff requires --device-ids")
            result = run_persistent_moe_uccl_ep_handoff(
                device_ids=args.device_ids,
                n=args.n,
                tensor_numel=args.tensor_numel,
                arch=args.arch,
                block_dim=args.block_dim,
                scheduler_blocks=args.scheduler_blocks,
                worker_blocks=args.worker_blocks,
                queue_capacity=args.queue_capacity,
                stream_id=args.stream_id,
                num_tokens=args.uccl_ep_num_tokens,
                num_topk=args.uccl_ep_num_topk,
                num_experts=args.uccl_ep_num_experts,
                input_dtype=args.uccl_ep_input_dtype,
                repeats=args.uccl_ep_repeats,
                uccl_ep_bench_dir=args.uccl_ep_bench_dir,
            )
        elif args.with_uccl_ep_fused_boundary:
            if args.device_ids is None:
                raise ValueError("--with-uccl-ep-fused-boundary requires --device-ids")
            result = run_persistent_moe_uccl_ep_fused_boundary(
                device_ids=args.device_ids,
                n=args.n,
                tensor_numel=args.tensor_numel,
                arch=args.arch,
                block_dim=args.block_dim,
                scheduler_blocks=args.scheduler_blocks,
                worker_blocks=args.worker_blocks,
                queue_capacity=args.queue_capacity,
                stream_id=args.stream_id,
                num_tokens=args.uccl_ep_num_tokens,
                num_topk=args.uccl_ep_num_topk,
                num_experts=args.uccl_ep_num_experts,
                input_dtype=args.uccl_ep_input_dtype,
                repeats=args.uccl_ep_repeats,
                uccl_ep_bench_dir=args.uccl_ep_bench_dir,
            )
        elif args.device_ids is None:
            result = run_moe_dispatch_combine(
                device=args.device,
                n=args.n,
                arch=args.arch,
                block_dim=args.block_dim,
                scheduler_blocks=args.scheduler_blocks,
                worker_blocks=args.worker_blocks,
                queue_capacity=args.queue_capacity,
                stream_id=args.stream_id,
            )
        else:
            result = run_two_device_moe_dispatch_combine(
                device_ids=args.device_ids,
                n=args.n,
                arch=args.arch,
                block_dim=args.block_dim,
                scheduler_blocks=args.scheduler_blocks,
                worker_blocks=args.worker_blocks,
                queue_capacity=args.queue_capacity,
                stream_id=args.stream_id,
            )
        result = {**result, "command": _display_command(raw_args)}
    except Exception as exc:
        result = {
            "schema_version": 1,
            "runtime": "persistent_device",
            "dag_shape": DAG_SHAPE,
            "status": "failed",
            "command": _display_command(raw_args),
            "error_type": type(exc).__name__,
            "error": _clean_text(str(exc)),
            "non_claims": [
                "failed runs are not CUDA correctness evidence",
                "no distributed, serving, DeepSeek, or performance claim",
            ],
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    if output_json is not None:
        if output_json.is_absolute():
            result = {
                **result,
                "status": "failed",
                "error_type": "ValueError",
                "error": "--output-json must be repo-relative",
            }
            text = json.dumps(result, indent=2, sort_keys=True)
        else:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(text + "\n", encoding="utf-8")
    print(text)

    if result["status"] == "failed":
        return 1
    if result["status"] == "skipped" and require_cuda:
        return 2
    if result["status"] == "unsupported":
        return 3
    return 0


def _relative_path(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(Path.cwd())
        except ValueError:
            path = Path(path.name)
    return path.as_posix()


def _clean_text(text: str) -> str:
    cwd = Path.cwd().as_posix()
    home = Path.home().as_posix()
    text = text.replace(cwd, ".").replace(home, "~")
    return re.sub(r"/tmp/pto-cu[-A-Za-z0-9_]*", "<tmp-checkout>", text)


def _review_safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<sanitized>"
            if key in {"uccl_ep_bench_dir"} and item
            else _relative_path(item)
            if key == "nccl_library"
            else _review_safe_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_review_safe_payload(item) for item in value]
    if isinstance(value, str):
        return _clean_text(value)
    return value


def _display_command(raw_args: list[str]) -> str:
    safe_args = []
    for arg in raw_args:
        path = Path(arg)
        safe_args.append(path.name if path.is_absolute() else arg)
    command = "examples/cuda/persistent_moe_dispatch_combine.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


def _last_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    payloads = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
        index = start + end
    return payloads[-1] if payloads else None


if __name__ == "__main__":
    raise SystemExit(main())
