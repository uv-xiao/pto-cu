#!/usr/bin/env python3
"""Run or describe a persistent-device MoE dispatch/combine graph."""

from __future__ import annotations

import argparse
import ctypes
import json
import shlex
import sys
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

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


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    output_json = DEFAULT_OUTPUT_JSON
    require_cuda = False
    try:
        parser = JsonArgumentParser(description=__doc__)
        parser.add_argument("--device", type=int, default=0)
        parser.add_argument("--n", type=int, default=4096)
        parser.add_argument("--arch", default="compute_80")
        parser.add_argument("--block-dim", type=int, default=256)
        parser.add_argument("--scheduler-blocks", type=int, default=1)
        parser.add_argument("--worker-blocks", type=int, default=4)
        parser.add_argument("--queue-capacity", type=int, default=5)
        parser.add_argument("--stream-id", type=int, default=0)
        parser.add_argument("--output-json", type=Path)
        parser.add_argument(
            "--require-cuda",
            action="store_true",
            help="return non-zero when CUDA tooling or a CUDA device is unavailable",
        )
        args = parser.parse_args(raw_args)
        output_json = args.output_json
        require_cuda = args.require_cuda
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
    return text.replace(cwd, ".").replace(home, "~")


def _display_command(raw_args: list[str]) -> str:
    safe_args = []
    for arg in raw_args:
        path = Path(arg)
        safe_args.append(path.name if path.is_absolute() else arg)
    command = "examples/cuda/persistent_moe_dispatch_combine.py"
    if safe_args:
        command = f"{command} {shlex.join(safe_args)}"
    return command


if __name__ == "__main__":
    raise SystemExit(main())
