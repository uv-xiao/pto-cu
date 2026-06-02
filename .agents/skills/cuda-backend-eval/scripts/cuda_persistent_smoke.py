#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone CUDA persistent_device smoke runner."""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda_smoke import PtoRunTiming, _bind_runtime, _find_nvcc

from cuda_persistent_smoke_impl.fallback_ptx import (
    FALLBACK_PERSISTENT_DAG_F32_PTX as _FALLBACK_PERSISTENT_DAG_F32_PTX,
    FALLBACK_PERSISTENT_QUEUE_VECTOR_ADD_PTX as _FALLBACK_PERSISTENT_QUEUE_VECTOR_ADD_PTX,
    FALLBACK_PERSISTENT_VECTOR_ADD_PTX as _FALLBACK_PERSISTENT_VECTOR_ADD_PTX,
)
from cuda_persistent_smoke_impl.normal_graph_shapes import (
    CPP_ORCHESTRATOR_SNAPSHOT_DAG_SHAPE,
    NORMAL_GRAPH_DAG_SHAPES,
    make_cpp_orchestrator_snapshot_submits,
    make_normal_graph_nodes,
)

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples" / "cuda"))

from qwen_persistent_task_bodies_impl.tensor_tiles import (  # noqa: E402
    qwen_tensor_tile_func_id,
    qwen_tensor_tile_task_functions,
)
from simpler_setup.cuda_callable_compiler import (
    CudaPersistentCallableArtifact,
    CudaPersistentTaskBodyFunction,
    CudaPersistentTaskFunction,
    CudaTaskBody,
    prepare_cuda_persistent_device_callable,
    render_persistent_dag_source,
)
from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDeviceCallable as CudaPersistentCallable,
)
from simpler_setup.cuda_normal_graph import CudaNormalGraphNode, lower_normal_graph
from simpler_setup.cuda_pto_graph import lower_cuda_pto_task_graph
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.runtime_builder import RuntimeBuilder

_PERSISTENT_DIRECT_SOURCE = """
struct PtoCudaPersistentVectorAddTask {
    const float *a;
    const float *b;
    float *out;
    unsigned long long n;
};

extern "C" __global__ void pto_persistent_vector_add_executor(
    const PtoCudaPersistentVectorAddTask *tasks, unsigned long long task_count) {
    unsigned long long task_id = blockIdx.x;
    if (task_id >= task_count) {
        return;
    }
    PtoCudaPersistentVectorAddTask task = tasks[task_id];
    for (unsigned long long i = threadIdx.x; i < task.n; i += blockDim.x) {
        task.out[i] = task.a[i] + task.b[i];
    }
}

extern "C" __global__ void pto_persistent_vector_add_grid_executor(
    const PtoCudaPersistentVectorAddTask *tasks,
    unsigned long long task_count,
    unsigned int worker_blocks_per_task) {
    unsigned long long task_id = blockIdx.x / worker_blocks_per_task;
    if (task_id >= task_count) {
        return;
    }
    unsigned int worker_block = blockIdx.x % worker_blocks_per_task;
    PtoCudaPersistentVectorAddTask task = tasks[task_id];
    unsigned long long first_i = threadIdx.x + static_cast<unsigned long long>(worker_block) * blockDim.x;
    unsigned long long stride = static_cast<unsigned long long>(blockDim.x) * worker_blocks_per_task;
    for (unsigned long long i = first_i; i < task.n; i += stride) {
        task.out[i] = task.a[i] + task.b[i];
    }
}
""".lstrip()

_PERSISTENT_QUEUE_SOURCE = """
struct PtoCudaPersistentVectorAddTask {
    const float *a;
    const float *b;
    float *out;
    unsigned long long n;
};

struct PtoCudaPersistentVectorAddQueueState {
    const PtoCudaPersistentVectorAddTask *tasks;
    unsigned long long task_count;
    unsigned int *ready_queue;
    unsigned int *ready_flags;
    unsigned int queue_capacity;
    unsigned int *queue_head;
    unsigned int *queue_tail;
    unsigned int *completed_count;
    unsigned int scheduler_blocks;
};

extern "C" __global__ void pto_persistent_vector_add_queue_executor(
    const PtoCudaPersistentVectorAddQueueState *state) {
    __shared__ unsigned int slot;
    __shared__ unsigned int task_id;

    unsigned int scheduler_blocks = state->scheduler_blocks == 0U ? 1U : state->scheduler_blocks;
    if (blockIdx.x < scheduler_blocks) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            for (unsigned long long task_id = 0; task_id < state->task_count; ++task_id) {
                unsigned int slot = static_cast<unsigned int>(task_id % state->queue_capacity);
                while (atomicAdd(&state->ready_flags[slot], 0U) != 0U) {
                }
                state->ready_queue[slot] = static_cast<unsigned int>(task_id);
                __threadfence();
                atomicExch(&state->ready_flags[slot], static_cast<unsigned int>(task_id + 1ULL));
                atomicAdd(state->queue_tail, 1U);
            }
        }
        return;
    }

    while (true) {
        if (threadIdx.x == 0) {
            slot = atomicAdd(state->queue_head, 1U);
            if (static_cast<unsigned long long>(slot) < state->task_count) {
                unsigned int ring_slot = slot % state->queue_capacity;
                unsigned int ready_value = slot + 1U;
                while (atomicAdd(&state->ready_flags[ring_slot], 0U) != ready_value) {
                }
                task_id = state->ready_queue[ring_slot];
                __threadfence();
                atomicExch(&state->ready_flags[ring_slot], 0U);
            }
        }
        __syncthreads();
        if (static_cast<unsigned long long>(slot) >= state->task_count) {
            break;
        }
        PtoCudaPersistentVectorAddTask task = state->tasks[task_id];
        for (unsigned long long i = threadIdx.x; i < task.n; i += blockDim.x) {
            task.out[i] = task.a[i] + task.b[i];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(state->completed_count, 1U);
        }
    }
}
""".lstrip()

_PERSISTENT_DAG_CONTEXT_DEFINITION = """
struct PtoTaskContext {
    const PtoCudaPersistentDagTask *task;
    unsigned long long i;
};
""".strip()

_PERSISTENT_DAG_TASK_FUNCTIONS = [
    CudaPersistentTaskBodyFunction(
        func_id=1,
        task_body=CudaTaskBody(
            name="add_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] + task->b[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=2,
        task_body=CudaTaskBody(
            name="mul_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->b[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=3,
        task_body=CudaTaskBody(
            name="matmul_tile_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
unsigned long long rows = static_cast<unsigned long long>(task->rows);
unsigned long long cols = static_cast<unsigned long long>(task->cols);
unsigned long long inner = static_cast<unsigned long long>(task->inner);
if (rows == 0ULL || cols == 0ULL || inner == 0ULL) {
  return;
}
unsigned long long matrix_elems = rows * cols;
unsigned long long tile_id = i / matrix_elems;
unsigned long long elem = i % matrix_elems;
unsigned long long row = elem / cols;
unsigned long long col = elem % cols;
unsigned long long a_base = tile_id * task->a_batch_stride;
unsigned long long b_base = tile_id * task->b_batch_stride;
unsigned long long out_base = tile_id * task->out_batch_stride;
float acc = 0.0f;
for (unsigned long long k = 0; k < inner; ++k) {
  acc += task->a[a_base + row * task->lda + k] * task->b[b_base + k * task->ldb + col];
}
task->out[out_base + row * task->ldc + col] = acc;
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=4,
        task_body=CudaTaskBody(
            name="axpy_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i] + task->b[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=11,
        task_body=CudaTaskBody(
            name="scale_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=5,
        task_body=CudaTaskBody(
            name="affine_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->scalar0 * task->a[i] + task->scalar1 * task->b[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=6,
        task_body=CudaTaskBody(
            name="triad_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->b[i] + task->c[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=7,
        task_body=CudaTaskBody(
            name="square_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->a[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=8,
        task_body=CudaTaskBody(
            name="quad_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
task->out[i] = task->a[i] * task->b[i] + task->c[i] * task->d[i];
""".strip(),
        ),
    ),
    CudaPersistentTaskBodyFunction(
        func_id=9,
        task_body=CudaTaskBody(
            name="generic_args_f32",
            context_definition=_PERSISTENT_DAG_CONTEXT_DEFINITION,
            body="""
const PtoCudaPersistentDagTask *task = ctx->task;
unsigned long long i = ctx->i;
if (task->tensor_arg_count < 2U || task->scalar_arg_count < 2U) {
  return;
}
task->out[i] = task->scalar_args[0] * task->a[i] +
               task->tensor_args[0][i] +
               task->scalar_args[1] * task->tensor_args[1][i];
if (task->tensor_arg_count >= 4U && task->scalar_arg_count >= 4U) {
  task->out[i] += task->scalar_args[2] * task->tensor_args[2][i] +
                  task->scalar_args[3] * task->tensor_args[3][i];
}
""".strip(),
        ),
    ),
    CudaPersistentTaskFunction(
        func_id=10,
        name="wmma_m16n16k8_f32",
        threading="block",
        body="""
if ((task->rows % 16U) != 0U || (task->cols % 16U) != 0U || task->inner == 0U || (task->inner % 8U) != 0U) {
  return;
}
using namespace nvcuda;
unsigned long long tile_count = task->n / task->out_batch_stride;
for (unsigned long long tile_id = 0; tile_id < tile_count; ++tile_id) {
  unsigned long long a_base = tile_id * task->a_batch_stride;
  unsigned long long b_base = tile_id * task->b_batch_stride;
  unsigned long long out_base = tile_id * task->out_batch_stride;
  for (unsigned int row = 0; row < task->rows; row += 16U) {
    for (unsigned int col = 0; col < task->cols; col += 16U) {
      wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
      wmma::fill_fragment(acc_frag, 0.0f);
      for (unsigned int k = 0; k < task->inner; k += 8U) {
        wmma::load_matrix_sync(a_frag, task->a + a_base + row * task->lda + k, task->lda);
        wmma::load_matrix_sync(b_frag, task->b + b_base + k * task->ldb + col, task->ldb);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      wmma::store_matrix_sync(task->out + out_base + row * task->ldc + col, acc_frag, task->ldc, wmma::mem_row_major);
    }
  }
}
""".strip(),
    ),
]
_PERSISTENT_DAG_TASK_FUNCTIONS.extend(qwen_tensor_tile_task_functions())
_PERSISTENT_DAG_SOURCE = render_persistent_dag_source(_PERSISTENT_DAG_TASK_FUNCTIONS)


def _tensor_core_func_id(descriptor: dict[str, int]) -> int:
    return qwen_tensor_tile_func_id(descriptor) or 10


def _compile_persistent_dag_ptx(work_dir: Path, arch: str) -> tuple[bytes, str, CudaPersistentCallableArtifact]:
    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc is required for persistent dag mode until the embedded PTX fallback is generated")

    task_sources = []
    for task in _PERSISTENT_DAG_TASK_FUNCTIONS:
        if isinstance(task, CudaPersistentTaskBodyFunction):
            source_path = work_dir / f"persistent_dag_{task.func_id}_{task.task_body.name}.pto.cu"
            source_path.write_text(task.task_body.body)
            task_sources.append(
                {
                    "func_id": task.func_id,
                    "task_name": task.task_body.name,
                    "source_path": str(source_path),
                    "body_style": "task_body",
                    "context_definition": task.task_body.context_definition,
                }
            )
        else:
            source_path = work_dir / f"persistent_dag_{task.func_id}_{task.name}.pto.cu"
            source_path.write_text(task.body)
            task_sources.append(
                {
                    "func_id": task.func_id,
                    "task_name": task.name,
                    "source_path": str(source_path),
                    "threading": task.threading,
                }
            )

    artifact = KernelCompiler(platform="cuda").compile_cuda_persistent_device(
        task_sources,
        arch=arch,
        nvcc=nvcc,
    )
    return artifact.ptx, f"nvcc-persistent-{artifact.source_kind}-{arch}", artifact


class CudaPersistentVectorAddTask(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("n", ctypes.c_uint64),
    ]


class CudaPersistentVectorAddArgs(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_void_p),
        ("task_count", ctypes.c_uint64),
    ]


class CudaPersistentVectorAddGridArgs(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_void_p),
        ("task_count", ctypes.c_uint64),
        ("worker_blocks_per_task", ctypes.c_uint32),
    ]


class CudaPersistentVectorAddQueueState(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_void_p),
        ("task_count", ctypes.c_uint64),
        ("ready_queue", ctypes.c_void_p),
        ("ready_flags", ctypes.c_void_p),
        ("queue_capacity", ctypes.c_uint32),
        ("queue_head", ctypes.c_void_p),
        ("queue_tail", ctypes.c_void_p),
        ("completed_count", ctypes.c_void_p),
        ("scheduler_blocks", ctypes.c_uint32),
    ]


class CudaPersistentVectorAddQueueArgs(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_void_p),
    ]


class CudaPersistentDagTask(ctypes.Structure):
    _fields_ = [
        ("func_id", ctypes.c_uint32),
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("n", ctypes.c_uint64),
        ("dependent_begin", ctypes.c_uint32),
        ("dependent_count", ctypes.c_uint32),
        ("initial_fanin", ctypes.c_uint32),
        ("scalar0", ctypes.c_float),
        ("scalar1", ctypes.c_float),
        ("rows", ctypes.c_uint32),
        ("cols", ctypes.c_uint32),
        ("inner", ctypes.c_uint32),
        ("lda", ctypes.c_uint32),
        ("ldb", ctypes.c_uint32),
        ("ldc", ctypes.c_uint32),
        ("a_batch_stride", ctypes.c_uint64),
        ("b_batch_stride", ctypes.c_uint64),
        ("out_batch_stride", ctypes.c_uint64),
        ("c", ctypes.c_void_p),
        ("d", ctypes.c_void_p),
        ("tensor_args", ctypes.c_void_p * 4),
        ("scalar_args", ctypes.c_float * 4),
        ("tensor_arg_count", ctypes.c_uint32),
        ("scalar_arg_count", ctypes.c_uint32),
    ]


class CudaPersistentDagState(ctypes.Structure):
    _fields_ = [
        ("tasks", ctypes.c_void_p),
        ("task_count", ctypes.c_uint64),
        ("dependents", ctypes.c_void_p),
        ("dependent_count", ctypes.c_uint64),
        ("fanin", ctypes.c_void_p),
        ("ready_queue", ctypes.c_void_p),
        ("ready_flags", ctypes.c_void_p),
        ("completion_queue", ctypes.c_void_p),
        ("completion_flags", ctypes.c_void_p),
        ("queue_capacity", ctypes.c_uint32),
        ("queue_head", ctypes.c_void_p),
        ("queue_tail", ctypes.c_void_p),
        ("completion_head", ctypes.c_void_p),
        ("completion_tail", ctypes.c_void_p),
        ("completed_count", ctypes.c_void_p),
        ("error_count", ctypes.c_void_p),
        ("error_code", ctypes.c_void_p),
        ("error_task_id", ctypes.c_void_p),
        ("scheduler_blocks", ctypes.c_uint32),
        ("scheduler_init_count", ctypes.c_void_p),
        ("scheduler_loop_count", ctypes.c_void_p),
        ("scheduler_processed_count", ctypes.c_void_p),
        ("scheduler_processed_by_block", ctypes.c_void_p),
    ]


class CudaPersistentDagArgs(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_void_p),
    ]


def _f32(value: float) -> float:
    return ctypes.c_float(value).value


def _fma_f32(a: float, b: float, c: float) -> float:
    return _f32(float(a) * float(b) + float(c))


def _first_mismatch(actual: list[float], expected: list[float], *, rtol: float = 0.0, atol: float = 0.0) -> str:
    if len(actual) != len(expected):
        return f"len actual={len(actual)} expected={len(expected)}"
    for idx, actual_value in enumerate(actual):
        expected_value = expected[idx]
        tolerance = atol + rtol * abs(expected_value)
        if abs(actual_value - expected_value) > tolerance:
            return f"idx={idx} actual={actual_value} expected={expected_value}"
    return "no mismatch"


def _make_tensor_tile_descriptor(rows: int = 16, cols: int = 16, inner: int = 16) -> dict[str, int]:
    if rows <= 0 or cols <= 0 or inner <= 0:
        raise ValueError("tensor tile rows, cols, and inner must be positive")
    return {
        "rows": rows,
        "cols": cols,
        "inner": inner,
        "lda": inner,
        "ldb": cols,
        "ldc": cols,
        "a_batch_stride": rows * inner,
        "b_batch_stride": inner * cols,
        "out_batch_stride": rows * cols,
    }


_TENSOR_TILE_DESCRIPTOR = _make_tensor_tile_descriptor()


def _is_tensor_tile_shape(dag_shape: str) -> bool:
    return dag_shape in {"graph_tensor_core_tile", "graph_tensor_tile", "tensor_core_tile", "tensor_tile"}


def _tensor_tile_buffer_lengths(n: int, descriptor: dict[str, int]) -> dict[str, int]:
    out_batch_stride = descriptor["out_batch_stride"]
    if n % out_batch_stride != 0:
        raise ValueError(f"tensor_tile DAG shape requires n to be a multiple of rows*cols ({out_batch_stride})")
    tile_count = n // out_batch_stride
    return {
        "a": max(n, tile_count * descriptor["a_batch_stride"]),
        "b": max(n, tile_count * descriptor["b_batch_stride"]),
        "output": n,
        "tile_count": tile_count,
    }


def _matmul_expected(host_a, host_b, n: int, descriptor: dict[str, int]) -> list[float]:
    expected: list[float] = [0.0] * n
    rows = descriptor["rows"]
    cols = descriptor["cols"]
    inner = descriptor["inner"]
    matrix_elems = rows * cols
    for i in range(n):
        tile_id = i // matrix_elems
        elem = i % matrix_elems
        row = elem // cols
        col = elem % cols
        a_base = tile_id * descriptor["a_batch_stride"]
        b_base = tile_id * descriptor["b_batch_stride"]
        out_base = tile_id * descriptor["out_batch_stride"]
        acc = 0.0
        for k in range(inner):
            a_index = a_base + row * descriptor["lda"] + k
            b_index = b_base + k * descriptor["ldb"] + col
            acc = _f32(acc + _f32(host_a[a_index] * host_b[b_index]))
        expected[out_base + row * descriptor["ldc"] + col] = acc
    return expected


@dataclass
class PersistentLaunch:
    manifest: CudaPersistentCallable
    args: Any
    scheduler_blocks: int
    worker_blocks: int
    completed_count: int
    host_counters: Any | None = None
    dev_counters: int | None = None
    host_ready_flags: Any | None = None
    dev_ready_flags: int | None = None
    host_fanin: Any | None = None
    dev_fanin: int | None = None
    dispatch_func_ids: list[int] | None = None
    device_allocations: tuple[int, ...] = ()


@dataclass
class DagSmokeConfig:
    runtime: Any
    binaries: Any
    ctx: int
    device: int
    n: int
    arch: str
    queue_capacity: int
    scheduler_blocks: int
    worker_blocks: int
    stream_id: int
    block_dim: int
    ptx: bytes
    ptx_buf: Any
    persistent_artifact: CudaPersistentCallableArtifact | None
    ptx_source: str
    start: float
    dag_shape: str
    tensor_tile: dict[str, int]
    repeat_runs: int


def _compile_persistent_ptx(
    work_dir: Path,
    arch: str,
    mode: str,
) -> tuple[bytes, str, CudaPersistentCallableArtifact | None]:
    nvcc = _find_nvcc()
    if nvcc is None:
        if mode == "direct":
            return _FALLBACK_PERSISTENT_VECTOR_ADD_PTX, "embedded-sm80-persistent-ptx", None
        if mode == "dag":
            if _FALLBACK_PERSISTENT_DAG_F32_PTX:
                return _FALLBACK_PERSISTENT_DAG_F32_PTX, "embedded-sm80-persistent-dag-ptx", None
            raise RuntimeError("nvcc is required for persistent dag mode until the embedded PTX fallback is generated")
        if _FALLBACK_PERSISTENT_QUEUE_VECTOR_ADD_PTX:
            return _FALLBACK_PERSISTENT_QUEUE_VECTOR_ADD_PTX, "embedded-sm80-persistent-queue-ptx", None
        raise RuntimeError("nvcc is required for persistent queue mode until the embedded PTX fallback is generated")

    source_by_mode = {
        "direct": _PERSISTENT_DIRECT_SOURCE,
        "queue": _PERSISTENT_QUEUE_SOURCE,
    }
    if mode == "dag":
        return _compile_persistent_dag_ptx(work_dir, arch)
    if mode not in source_by_mode:
        raise ValueError(f"unknown persistent mode: {mode}")

    kernel_src = work_dir / f"persistent_vector_add_{mode}.cu"
    kernel_src.write_text(source_by_mode[mode])
    ptx_path = work_dir / "persistent_vector_add.ptx"
    subprocess.run(
        [nvcc, "--ptx", "-std=c++17", f"-arch={arch}", str(kernel_src), "-o", str(ptx_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    source_kind = "generated-dispatch" if mode == "dag" else mode
    return ptx_path.read_bytes(), f"nvcc-persistent-{source_kind}-{arch}", None


def _make_dag_shape(  # noqa: PLR0912, PLR0915
    dag_shape: str,
    n: int,
    dev_a: int,
    dev_b: int,
    dev_tmp0: int,
    dev_tmp1: int,
    dev_tmp2: int,
    dev_tmp3: int,
    dev_out: int,
    tensor_tile: dict[str, int] | None = None,
) -> tuple[Any, Any, Any]:
    def make_task(
        func_id: int,
        a: int,
        b: int,
        out: int,
        dependent_begin: int,
        dependent_count: int,
        initial_fanin: int,
        descriptor: dict[str, int] | None = None,
    ) -> CudaPersistentDagTask:
        task = CudaPersistentDagTask(
            func_id,
            a,
            b,
            out,
            n,
            dependent_begin,
            dependent_count,
            initial_fanin,
        )
        if descriptor is not None:
            for field, value in descriptor.items():
                setattr(task, field, value)
        return task

    if dag_shape == CPP_ORCHESTRATOR_SNAPSHOT_DAG_SHAPE:
        submits = make_cpp_orchestrator_snapshot_submits(
            n=n,
            dev_a=dev_a,
            dev_b=dev_b,
            dev_tmp1=dev_tmp1,
            dev_out=dev_out,
        )
        lowered = lower_cuda_pto_task_graph(
            submits,
            lambda node, dependent_begin, dependent_count, initial_fanin: make_task(
                int(node.attrs["submit"].attrs["cuda_task"]["func_id"]),
                int(node.attrs["submit"].attrs["cuda_task"]["a"]),
                int(node.attrs["submit"].attrs["cuda_task"]["b"]),
                int(node.attrs["submit"].attrs["cuda_task"]["out"]),
                dependent_begin,
                dependent_count,
                initial_fanin,
            ),
        )
        host_fanin_t = ctypes.c_uint32 * len(lowered.fanin)
        dependents_t = ctypes.c_uint32 * len(lowered.dependents)
        task_t = CudaPersistentDagTask * len(lowered.tasks)
        return (
            host_fanin_t(*lowered.fanin),
            dependents_t(*lowered.dependents),
            task_t(*lowered.tasks),
        )

    if dag_shape in NORMAL_GRAPH_DAG_SHAPES:
        nodes = make_normal_graph_nodes(
            dag_shape,
            n=n,
            dev_a=dev_a,
            dev_b=dev_b,
            dev_tmp0=dev_tmp0,
            dev_tmp1=dev_tmp1,
            dev_tmp2=dev_tmp2,
            dev_tmp3=dev_tmp3,
            dev_out=dev_out,
        )
        lowered = lower_normal_graph(
            nodes,
            lambda node, dependent_begin, dependent_count, initial_fanin: make_task(
                node.func_id,
                node.a,
                node.b,
                node.out,
                dependent_begin,
                dependent_count,
                initial_fanin,
                node.attrs,
            ),
        )
        host_fanin_t = ctypes.c_uint32 * len(lowered.fanin)
        dependents_t = ctypes.c_uint32 * len(lowered.dependents)
        task_t = CudaPersistentDagTask * len(lowered.tasks)
        return (
            host_fanin_t(*lowered.fanin),
            dependents_t(*lowered.dependents),
            task_t(*lowered.tasks),
        )

    if dag_shape == "fork_join":
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {"chain", "graph_descriptor_chain"}:
        task_count = 5
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 4
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2, 1, 1),
            dependents_t(2, 2, 3, 4),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=2,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp2,
                    b=dev_b,
                    out=dev_tmp3,
                    n=n,
                    dependent_begin=3,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp2,
                    b=dev_tmp3,
                    out=dev_out,
                    n=n,
                    dependent_begin=4,
                    dependent_count=0,
                    initial_fanin=1,
                ),
            ),
        )
    if dag_shape in {"scratch_reuse", "graph_descriptor_scratch_reuse"}:
        task_count = 6
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 6
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2, 1, 1, 2),
            dependents_t(2, 2, 3, 4, 5, 5),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=2,
                    dependent_count=2,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp2,
                    b=dev_b,
                    out=dev_tmp3,
                    n=n,
                    dependent_begin=4,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp2,
                    b=dev_a,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=5,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp3,
                    out=dev_out,
                    n=n,
                    dependent_begin=6,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_parallel_chains":
        task_count = 9
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 10
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 0, 0, 2, 2, 2, 2, 2),
            dependents_t(4, 4, 5, 5, 6, 7, 6, 7, 8, 8),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=2,
                    dependent_count=1,
                    initial_fanin=0,
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
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=4,
                    dependent_count=2,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp2,
                    b=dev_tmp3,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=6,
                    dependent_count=2,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp0,
                    b=dev_tmp2,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=8,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp2,
                    out=dev_tmp3,
                    n=n,
                    dependent_begin=9,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp3,
                    out=dev_out,
                    n=n,
                    dependent_begin=10,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_wide_fanout":
        task_count = 7
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 9
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 1, 1, 1, 2, 2, 2),
            dependents_t(1, 2, 3, 4, 4, 5, 5, 6, 6),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=3,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_a,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=3,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp0,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=4,
                    dependent_count=2,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_b,
                    out=dev_tmp3,
                    n=n,
                    dependent_begin=6,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=7,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp2,
                    b=dev_tmp3,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=8,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp0,
                    out=dev_out,
                    n=n,
                    dependent_begin=9,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_multi_fanin":
        task_count = 4
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 3
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 0, 3),
            dependents_t(3, 3, 3),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=11,
                    a=dev_a,
                    b=0,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=2,
                    dependent_count=1,
                    initial_fanin=0,
                    scalar0=2.0,
                ),
                CudaPersistentDagTask(
                    func_id=6,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    c=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=3,
                    dependent_count=0,
                    initial_fanin=3,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_layered_cross":
        task_count = 9
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 13
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 0, 2, 3, 1, 2, 3, 2),
            dependents_t(3, 3, 4, 4, 5, 4, 6, 7, 6, 7, 7, 8, 8),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=2,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=11,
                    a=dev_a,
                    b=0,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=3,
                    dependent_count=2,
                    initial_fanin=0,
                    scalar0=2.0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_tmp3,
                    n=n,
                    dependent_begin=5,
                    dependent_count=3,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=8,
                    dependent_count=2,
                    initial_fanin=3,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp2,
                    b=dev_a,
                    out=dev_out,
                    n=n,
                    dependent_begin=10,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=6,
                    a=dev_tmp3,
                    b=dev_tmp0,
                    c=dev_a,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=11,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp3,
                    b=dev_out,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=12,
                    dependent_count=1,
                    initial_fanin=3,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=13,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {"graph_tensor_core_tile", "graph_tensor_tile", "tensor_core_tile", "tensor_tile"}:
        descriptor = tensor_tile or _TENSOR_TILE_DESCRIPTOR
        task_count = 4
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 4
        task_t = CudaPersistentDagTask * task_count
        if dag_shape in {"graph_tensor_core_tile", "tensor_core_tile"}:
            func_id = _tensor_core_func_id(descriptor)
        else:
            func_id = 3
        return (
            host_fanin_t(0, 1, 1, 2),
            dependents_t(1, 2, 3, 3),
            task_t(
                make_task(
                    func_id,
                    dev_a,
                    dev_b,
                    dev_tmp0,
                    0,
                    2,
                    0,
                    descriptor,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_a,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=2,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp0,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=3,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=4,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {"scalar_axpy", "graph_descriptor_scalar_axpy"}:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=4,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    scalar0=1.5,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {"scalar_scale", "graph_descriptor_scalar_scale"}:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=11,
                    a=dev_a,
                    b=0,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    scalar0=2.0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {"scalar_affine", "graph_descriptor_scalar_affine"}:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=5,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    scalar0=1.5,
                    scalar1=0.5,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {"triad", "graph_descriptor_triad"}:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=6,
                    a=dev_a,
                    b=dev_b,
                    c=dev_tmp0,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {"unary_square", "graph_descriptor_unary_square"}:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 1, 1),
            dependents_t(1, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=7,
                    a=dev_a,
                    b=0,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_a,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=1,
                ),
            ),
        )
    if dag_shape in {"quad", "graph_descriptor_quad"}:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=8,
                    a=dev_a,
                    b=dev_b,
                    c=dev_tmp0,
                    d=dev_tmp3,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_reordered":
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        tensor_args_t = ctypes.c_void_p * 4
        scalar_args_t = ctypes.c_float * 4
        return (
            host_fanin_t(2, 0, 0),
            dependents_t(0, 0),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=0,
                    dependent_count=0,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=9,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    tensor_args=tensor_args_t(dev_tmp0, dev_tmp3, 0, 0),
                    scalar_args=scalar_args_t(1.5, 0.25, 0.0, 0.0),
                    tensor_arg_count=2,
                    scalar_arg_count=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
            ),
        )
    if dag_shape in {
        "graph_descriptor_depends_on",
        "graph_descriptor_node_io",
        "graph_descriptor_node_link",
        "graph_descriptor_named_callable",
        "graph_descriptor_node_op",
        "graph_descriptor_node_port_dict",
        "graph_descriptor_task_dict",
    }:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_diamond":
        task_count = 5
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 6
        task_t = CudaPersistentDagTask * task_count
        tensor_args_t = ctypes.c_void_p * 4
        scalar_args_t = ctypes.c_float * 4
        return (
            host_fanin_t(0, 0, 2, 2, 2),
            dependents_t(2, 3, 2, 3, 4, 4),
            task_t(
                CudaPersistentDagTask(
                    func_id=9,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=0,
                    dependent_count=2,
                    initial_fanin=0,
                    tensor_args=tensor_args_t(dev_tmp0, dev_tmp3, 0, 0),
                    scalar_args=scalar_args_t(1.5, 0.25, 0.0, 0.0),
                    tensor_arg_count=2,
                    scalar_arg_count=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=2,
                    dependent_count=2,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=4,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_tmp3,
                    n=n,
                    dependent_begin=5,
                    dependent_count=1,
                    initial_fanin=2,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp3,
                    out=dev_out,
                    n=n,
                    dependent_begin=6,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_submits":
        nodes = (
            CudaNormalGraphNode("task0", func_id=1, a=dev_a, b=dev_b, out=dev_tmp1, n=n),
            CudaNormalGraphNode(
                "task1",
                depends_on=("task0",),
                func_id=1,
                a=dev_tmp1,
                b=dev_b,
                out=dev_tmp1,
                n=n,
            ),
            CudaNormalGraphNode(
                "task2",
                depends_on=("task1",),
                func_id=1,
                a=dev_tmp1,
                b=dev_a,
                out=dev_out,
                n=n,
            ),
        )
        lowered = lower_normal_graph(
            nodes,
            lambda node, dependent_begin, dependent_count, initial_fanin: make_task(
                node.func_id,
                node.a,
                node.b,
                node.out,
                dependent_begin,
                dependent_count,
                initial_fanin,
                node.attrs,
            ),
        )
        host_fanin_t = ctypes.c_uint32 * len(lowered.fanin)
        dependents_t = ctypes.c_uint32 * len(lowered.dependents)
        task_t = CudaPersistentDagTask * len(lowered.tasks)
        return (
            host_fanin_t(*lowered.fanin),
            dependents_t(*lowered.dependents),
            task_t(*lowered.tasks),
        )
    if dag_shape in {
        "graph_descriptor_compact_role_inout",
        "graph_descriptor_pair_inout",
        "graph_descriptor_role_map_inout",
        "graph_descriptor_tagged_inout",
        "graph_descriptor_role_keyed_inout",
    }:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 1, 1),
            dependents_t(1, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=1,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_a,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=1,
                ),
            ),
        )
    if dag_shape == "graph_descriptor_submit_groups":
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape in {
        "generic_args",
        "generic_args4",
        "graph_descriptor",
        "graph_descriptor_generic_args4",
        "graph_descriptor_node_attrs",
        "graph_descriptor_tagged",
    }:
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        tensor_args_t = ctypes.c_void_p * 4
        scalar_args_t = ctypes.c_float * 4
        if dag_shape in {"generic_args4", "graph_descriptor_generic_args4"}:
            tensor_args = tensor_args_t(dev_tmp0, dev_tmp3, dev_a, dev_b)
            scalar_args = scalar_args_t(1.5, 0.25, 0.125, 0.0625)
            tensor_arg_count = 4
            scalar_arg_count = 4
        else:
            tensor_args = tensor_args_t(dev_tmp0, dev_tmp3, 0, 0)
            scalar_args = scalar_args_t(1.5, 0.25, 0.0, 0.0)
            tensor_arg_count = 2
            scalar_arg_count = 2
        return (
            host_fanin_t(0, 0, 2),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=9,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                    tensor_args=tensor_args,
                    scalar_args=scalar_args,
                    tensor_arg_count=tensor_arg_count,
                    scalar_arg_count=scalar_arg_count,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp2,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp1,
                    b=dev_tmp2,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "bad_func_id":
        task_count = 1
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 1
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0),
            dependents_t(0),
            task_t(
                CudaPersistentDagTask(
                    func_id=99,
                    a=dev_a,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=0,
                    dependent_count=0,
                    initial_fanin=0,
                )
            ),
        )
    if dag_shape == "bad_dependent":
        task_count = 1
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 1
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0),
            dependents_t(7),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                )
            ),
        )
    if dag_shape == "bad_dependent_range":
        task_count = 1
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 1
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0),
            dependents_t(0),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                )
            ),
        )
    if dag_shape == "bad_fanin_underflow":
        task_count = 3
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 0, 1),
            dependents_t(2, 2),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=2,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp1,
                    n=n,
                    dependent_begin=1,
                    dependent_count=1,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_tmp1,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=1,
                ),
            ),
        )
    if dag_shape == "bad_duplicate_dependent":
        task_count = 2
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 2
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 2),
            dependents_t(1, 1),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=2,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=2,
                    dependent_count=0,
                    initial_fanin=2,
                ),
            ),
        )
    if dag_shape == "bad_self_dependent":
        task_count = 1
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 1
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0),
            dependents_t(0),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=0,
                    dependent_count=1,
                    initial_fanin=0,
                )
            ),
        )
    if dag_shape == "bad_initial_fanin":
        task_count = 1
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 1
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(1),
            dependents_t(0),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=0,
                    dependent_count=0,
                    initial_fanin=0,
                )
            ),
        )
    if dag_shape == "bad_no_root":
        task_count = 1
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 1
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(1),
            dependents_t(0),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=0,
                    dependent_count=0,
                    initial_fanin=1,
                )
            ),
        )
    if dag_shape == "bad_unreachable":
        task_count = 2
        host_fanin_t = ctypes.c_uint32 * task_count
        dependents_t = ctypes.c_uint32 * 1
        task_t = CudaPersistentDagTask * task_count
        return (
            host_fanin_t(0, 1),
            dependents_t(0),
            task_t(
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_a,
                    b=dev_b,
                    out=dev_tmp0,
                    n=n,
                    dependent_begin=0,
                    dependent_count=0,
                    initial_fanin=0,
                ),
                CudaPersistentDagTask(
                    func_id=1,
                    a=dev_tmp0,
                    b=dev_b,
                    out=dev_out,
                    n=n,
                    dependent_begin=0,
                    dependent_count=0,
                    initial_fanin=1,
                ),
            ),
        )
    raise ValueError(f"unknown dag shape: {dag_shape}")


@functools.lru_cache(maxsize=1)
def _load_persistent_runtime():
    binaries = RuntimeBuilder(platform="cuda").get_binaries("persistent_device", build=True)
    runtime = ctypes.CDLL(str(binaries.host_path))
    _bind_runtime(runtime)
    return runtime, binaries


def _make_direct_launch(
    ptx_buf,
    ptx_size: int,
    task_count: int,
    dev_tasks: int,
    worker_blocks_per_task: int = 1,
    stream_id: int = 0,
    block_dim: int = 256,
) -> PersistentLaunch:
    if worker_blocks_per_task <= 0:
        raise ValueError("worker_blocks_per_task must be positive")
    worker_blocks = max(1, task_count * worker_blocks_per_task)
    entry_name = b"pto_persistent_vector_add_executor"
    op = 1001
    args: Any = CudaPersistentVectorAddArgs(tasks=dev_tasks, task_count=task_count)
    if worker_blocks_per_task > 1:
        entry_name = b"pto_persistent_vector_add_grid_executor"
        op = 1004
        args = CudaPersistentVectorAddGridArgs(
            tasks=dev_tasks,
            task_count=task_count,
            worker_blocks_per_task=worker_blocks_per_task,
        )
    return PersistentLaunch(
        manifest=CudaPersistentCallable(
            version=2,
            op=op,
            image=ctypes.cast(ptx_buf, ctypes.c_void_p),
            image_size=ptx_size,
            entry_name=entry_name,
            grid_dim=worker_blocks,
            block_dim=block_dim,
            shared_mem_bytes=0,
            stream_id=stream_id,
        ),
        args=args,
        scheduler_blocks=0,
        worker_blocks=worker_blocks,
        completed_count=task_count,
    )


def _make_queue_launch(  # noqa: PLR0913
    runtime,
    ctx: int,
    ptx_buf,
    ptx_size: int,
    task_count: int,
    queue_capacity: int,
    dev_tasks: int,
    scheduler_blocks: int,
    worker_blocks: int,
    stream_id: int = 0,
    block_dim: int = 256,
) -> PersistentLaunch:
    host_counters_t = ctypes.c_uint32 * 3
    host_counters = host_counters_t(0, 0, 0)
    host_flags_t = ctypes.c_uint32 * queue_capacity
    host_flags = host_flags_t(*([0] * queue_capacity))
    dev_ready_queue = runtime.device_malloc_ctx(ctx, ctypes.sizeof(ctypes.c_uint32 * queue_capacity))
    dev_ready_flags = runtime.device_malloc_ctx(ctx, ctypes.sizeof(host_flags))
    dev_counters = runtime.device_malloc_ctx(ctx, ctypes.sizeof(host_counters))
    dev_queue_state = runtime.device_malloc_ctx(ctx, ctypes.sizeof(CudaPersistentVectorAddQueueState))
    device_allocations = tuple(ptr for ptr in (dev_ready_queue, dev_ready_flags, dev_counters, dev_queue_state) if ptr)
    try:
        if not (dev_ready_queue and dev_ready_flags and dev_counters and dev_queue_state):
            raise RuntimeError("queue device allocation failed")
        if runtime.copy_to_device_ctx(ctx, dev_ready_flags, ctypes.byref(host_flags), ctypes.sizeof(host_flags)) != 0:
            raise RuntimeError("copy_to_device queue flags failed")
        if (
            runtime.copy_to_device_ctx(ctx, dev_counters, ctypes.byref(host_counters), ctypes.sizeof(host_counters))
            != 0
        ):
            raise RuntimeError("copy_to_device queue counters failed")
        queue_state = CudaPersistentVectorAddQueueState(
            tasks=dev_tasks,
            task_count=task_count,
            ready_queue=dev_ready_queue,
            ready_flags=dev_ready_flags,
            queue_capacity=queue_capacity,
            queue_head=dev_counters,
            queue_tail=dev_counters + ctypes.sizeof(ctypes.c_uint32),
            completed_count=dev_counters + 2 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_blocks=scheduler_blocks,
        )
        if runtime.copy_to_device_ctx(ctx, dev_queue_state, ctypes.byref(queue_state), ctypes.sizeof(queue_state)) != 0:
            raise RuntimeError("copy_to_device queue state failed")
    except Exception:
        for ptr in device_allocations:
            runtime.device_free_ctx(ctx, ptr)
        raise

    return PersistentLaunch(
        manifest=CudaPersistentCallable(
            version=2,
            op=1002,
            image=ctypes.cast(ptx_buf, ctypes.c_void_p),
            image_size=ptx_size,
            entry_name=b"pto_persistent_vector_add_queue_executor",
            grid_dim=scheduler_blocks + worker_blocks,
            block_dim=block_dim,
            shared_mem_bytes=0,
            stream_id=stream_id,
        ),
        args=CudaPersistentVectorAddQueueArgs(state=dev_queue_state),
        scheduler_blocks=scheduler_blocks,
        worker_blocks=worker_blocks,
        completed_count=0,
        host_counters=host_counters,
        dev_counters=dev_counters,
        host_ready_flags=host_flags,
        dev_ready_flags=dev_ready_flags,
        device_allocations=device_allocations,
    )


def _make_launch(  # noqa: PLR0913
    runtime,
    ctx: int,
    mode: str,
    ptx_buf,
    ptx_size: int,
    task_count: int,
    queue_capacity: int,
    dev_tasks: int,
    worker_blocks_per_task: int = 1,
    worker_blocks: int | None = None,
    scheduler_blocks: int = 1,
    stream_id: int = 0,
    block_dim: int = 256,
) -> PersistentLaunch:
    if mode == "direct":
        return _make_direct_launch(
            ptx_buf,
            ptx_size,
            task_count,
            dev_tasks,
            worker_blocks_per_task,
            stream_id,
            block_dim,
        )
    return _make_queue_launch(
        runtime,
        ctx,
        ptx_buf,
        ptx_size,
        task_count,
        queue_capacity,
        dev_tasks,
        scheduler_blocks,
        worker_blocks if worker_blocks is not None else max(1, task_count),
        stream_id,
        block_dim,
    )


def _run_dag_smoke(config: DagSmokeConfig) -> dict:  # noqa: PLR0912, PLR0915
    runtime = config.runtime
    ctx = config.ctx
    n = config.n
    queue_capacity = config.queue_capacity
    tensor_lengths = None
    if _is_tensor_tile_shape(config.dag_shape):
        tensor_lengths = _tensor_tile_buffer_lengths(n, config.tensor_tile)
        a_len = tensor_lengths["a"]
        b_len = tensor_lengths["b"]
        output_len = tensor_lengths["output"]
    else:
        a_len = n
        b_len = n
        output_len = n
    array_a_t = ctypes.c_float * a_len
    array_b_t = ctypes.c_float * b_len
    array_t = ctypes.c_float * output_len
    if _is_tensor_tile_shape(config.dag_shape):
        host_a = array_a_t(*[float((i % 5) + 1) for i in range(a_len)])
        host_b = array_b_t(*[float((i % 3) + 1) for i in range(b_len)])
    else:
        host_a = array_a_t(*[float(i) for i in range(a_len)])
        host_b = array_b_t(*[float(2 * i) for i in range(b_len)])
    graph_arg_shapes = {
        "generic_args",
        "generic_args4",
        "graph_descriptor",
        "graph_descriptor_diamond",
        "graph_descriptor_generic_args4",
        "graph_descriptor_node_attrs",
        "graph_descriptor_reordered",
        "graph_descriptor_tagged",
    }
    triad_shapes = {"triad", "graph_descriptor_triad"}
    quad_shapes = {"quad", "graph_descriptor_quad"}
    if config.dag_shape in {*triad_shapes, *quad_shapes, *graph_arg_shapes}:
        host_tmp0_seed = array_t(*[float(3 * i) for i in range(output_len)])
        host_tmp0 = array_t(*host_tmp0_seed)
    else:
        host_tmp0_seed = None
        host_tmp0 = array_t()
    host_tmp1 = array_t()
    host_tmp2 = array_t()
    if config.dag_shape in {*quad_shapes, *graph_arg_shapes}:
        host_tmp3_seed = array_t(*[float(4 * i) for i in range(output_len)])
        host_tmp3 = array_t(*host_tmp3_seed)
    else:
        host_tmp3_seed = None
        host_tmp3 = array_t()
    host_out = array_t()
    a_nbytes = ctypes.sizeof(host_a)
    b_nbytes = ctypes.sizeof(host_b)
    output_nbytes = ctypes.sizeof(host_out)

    scheduler_blocks = config.scheduler_blocks
    worker_blocks = config.worker_blocks

    dev_a = runtime.device_malloc_ctx(ctx, a_nbytes)
    dev_b = runtime.device_malloc_ctx(ctx, b_nbytes)
    dev_tmp0 = runtime.device_malloc_ctx(ctx, output_nbytes)
    dev_tmp1 = runtime.device_malloc_ctx(ctx, output_nbytes)
    dev_tmp2 = runtime.device_malloc_ctx(ctx, output_nbytes)
    dev_tmp3 = runtime.device_malloc_ctx(ctx, output_nbytes)
    dev_out = runtime.device_malloc_ctx(ctx, output_nbytes)
    host_counters_t = ctypes.c_uint32 * 11
    host_counters = host_counters_t(*([0] * 11))
    host_flags_t = ctypes.c_uint32 * queue_capacity
    host_flags = host_flags_t(*([0] * queue_capacity))
    host_fanin, dependents, tasks = _make_dag_shape(
        config.dag_shape,
        n,
        dev_a,
        dev_b,
        dev_tmp0,
        dev_tmp1,
        dev_tmp2,
        dev_tmp3,
        dev_out,
        tensor_tile=config.tensor_tile,
    )
    task_count = len(tasks)

    dev_tasks = runtime.device_malloc_ctx(ctx, ctypes.sizeof(tasks))
    dev_dependents = runtime.device_malloc_ctx(ctx, ctypes.sizeof(dependents))
    dev_fanin = runtime.device_malloc_ctx(ctx, ctypes.sizeof(host_fanin))
    dev_ready_queue = runtime.device_malloc_ctx(ctx, ctypes.sizeof(ctypes.c_uint32 * queue_capacity))
    dev_ready_flags = runtime.device_malloc_ctx(ctx, ctypes.sizeof(host_flags))
    dev_completion_queue = runtime.device_malloc_ctx(ctx, ctypes.sizeof(ctypes.c_uint32 * queue_capacity))
    dev_completion_flags = runtime.device_malloc_ctx(ctx, ctypes.sizeof(host_flags))
    dev_counters = runtime.device_malloc_ctx(ctx, ctypes.sizeof(host_counters))
    host_scheduler_processed_by_block_t = ctypes.c_uint32 * scheduler_blocks
    host_scheduler_processed_by_block = host_scheduler_processed_by_block_t(*([0] * scheduler_blocks))
    dev_scheduler_processed_by_block = runtime.device_malloc_ctx(ctx, ctypes.sizeof(host_scheduler_processed_by_block))
    dev_state = runtime.device_malloc_ctx(ctx, ctypes.sizeof(CudaPersistentDagState))
    allocated = (
        dev_a,
        dev_b,
        dev_tmp0,
        dev_tmp1,
        dev_tmp2,
        dev_tmp3,
        dev_out,
        dev_tasks,
        dev_dependents,
        dev_fanin,
        dev_ready_queue,
        dev_ready_flags,
        dev_completion_queue,
        dev_completion_flags,
        dev_counters,
        dev_scheduler_processed_by_block,
        dev_state,
    )
    if not all(allocated):
        raise RuntimeError("dag device allocation failed")

    try:
        static_copies = [
            (dev_a, ctypes.byref(host_a), a_nbytes, "a"),
            (dev_b, ctypes.byref(host_b), b_nbytes, "b"),
            (dev_tasks, ctypes.byref(tasks), ctypes.sizeof(tasks), "tasks"),
            (dev_dependents, ctypes.byref(dependents), ctypes.sizeof(dependents), "dependents"),
        ]
        for dst, src, size, label in static_copies:
            if runtime.copy_to_device_ctx(ctx, dst, src, size) != 0:
                raise RuntimeError(f"copy_to_device dag {label} failed")

        state = CudaPersistentDagState(
            tasks=dev_tasks,
            task_count=task_count,
            dependents=dev_dependents,
            dependent_count=len(dependents),
            fanin=dev_fanin,
            ready_queue=dev_ready_queue,
            ready_flags=dev_ready_flags,
            completion_queue=dev_completion_queue,
            completion_flags=dev_completion_flags,
            queue_capacity=queue_capacity,
            queue_head=dev_counters,
            queue_tail=dev_counters + ctypes.sizeof(ctypes.c_uint32),
            completion_head=dev_counters + 2 * ctypes.sizeof(ctypes.c_uint32),
            completion_tail=dev_counters + 3 * ctypes.sizeof(ctypes.c_uint32),
            completed_count=dev_counters + 4 * ctypes.sizeof(ctypes.c_uint32),
            error_count=dev_counters + 5 * ctypes.sizeof(ctypes.c_uint32),
            error_code=dev_counters + 6 * ctypes.sizeof(ctypes.c_uint32),
            error_task_id=dev_counters + 7 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_blocks=config.scheduler_blocks,
            scheduler_init_count=dev_counters + 8 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_loop_count=dev_counters + 9 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_processed_count=dev_counters + 10 * ctypes.sizeof(ctypes.c_uint32),
            scheduler_processed_by_block=dev_scheduler_processed_by_block,
        )
        if runtime.copy_to_device_ctx(ctx, dev_state, ctypes.byref(state), ctypes.sizeof(state)) != 0:
            raise RuntimeError("copy_to_device dag state failed")

        initial_fanin = [int(value) for value in host_fanin]
        zero_output = array_t()

        def reset_launch_state() -> tuple[Any, Any, Any]:
            launch_fanin_t = ctypes.c_uint32 * len(initial_fanin)
            launch_fanin = launch_fanin_t(*initial_fanin)
            launch_flags = host_flags_t(*([0] * queue_capacity))
            launch_counters = host_counters_t(*([0] * 11))
            launch_scheduler_processed_by_block = host_scheduler_processed_by_block_t(*([0] * scheduler_blocks))
            reset_copies = [
                (dev_fanin, ctypes.byref(launch_fanin), ctypes.sizeof(launch_fanin), "fanin"),
                (dev_ready_flags, ctypes.byref(launch_flags), ctypes.sizeof(launch_flags), "ready flags"),
                (
                    dev_completion_flags,
                    ctypes.byref(launch_flags),
                    ctypes.sizeof(launch_flags),
                    "completion flags",
                ),
                (dev_counters, ctypes.byref(launch_counters), ctypes.sizeof(launch_counters), "counters"),
                (
                    dev_scheduler_processed_by_block,
                    ctypes.byref(launch_scheduler_processed_by_block),
                    ctypes.sizeof(launch_scheduler_processed_by_block),
                    "scheduler processed by block",
                ),
                (dev_tmp1, ctypes.byref(zero_output), output_nbytes, "tmp1"),
                (dev_tmp2, ctypes.byref(zero_output), output_nbytes, "tmp2"),
                (dev_out, ctypes.byref(zero_output), output_nbytes, "out"),
            ]
            if config.dag_shape in {*triad_shapes, *quad_shapes, *graph_arg_shapes}:
                reset_tmp0 = host_tmp0_seed if host_tmp0_seed is not None else host_tmp0
                reset_copies.append((dev_tmp0, ctypes.byref(reset_tmp0), output_nbytes, "tmp0/c"))
            else:
                reset_copies.append((dev_tmp0, ctypes.byref(zero_output), output_nbytes, "tmp0"))
            if config.dag_shape in {*quad_shapes, *graph_arg_shapes}:
                reset_tmp3 = host_tmp3_seed if host_tmp3_seed is not None else host_tmp3
                reset_copies.append((dev_tmp3, ctypes.byref(reset_tmp3), output_nbytes, "tmp3/d"))
            else:
                reset_copies.append((dev_tmp3, ctypes.byref(zero_output), output_nbytes, "tmp3"))
            for dst, src, size, label in reset_copies:
                if runtime.copy_to_device_ctx(ctx, dst, src, size) != 0:
                    raise RuntimeError(f"copy_to_device dag {label} failed")
            return launch_fanin, launch_counters, launch_scheduler_processed_by_block

        artifact = config.persistent_artifact or CudaPersistentCallableArtifact(
            cache_key="embedded",
            cache_hit=True,
            source_path=Path(),
            ptx_path=Path(),
            manifest_path=Path(),
            ptx=config.ptx,
            entry_name="pto_persistent_dag_f32_executor",
            arch=config.arch,
            source_kind="generated-dispatch",
        )
        prepared = prepare_cuda_persistent_device_callable(
            artifact,
            op=1003,
            grid_dim=scheduler_blocks + worker_blocks,
            block_dim=config.block_dim,
            shared_mem_bytes=0,
            stream_id=config.stream_id,
        )
        args = CudaPersistentDagArgs(state=dev_state)
        if runtime.prepare_callable(ctx, 0, prepared.byref()) != 0:
            raise RuntimeError("prepare_callable dag failed")
        launch_completed_counts = []
        launch_host_wall_ns = []
        launch_device_wall_ns = []
        completed_count = 0
        error_count = 0
        error_code = 0
        error_task_id = 0
        scheduler_init_count = 0
        scheduler_loop_count = 0
        scheduler_processed_count = 0
        scheduler_processed_by_block: list[int] = []
        for launch_idx in range(config.repeat_runs):
            host_fanin, host_counters, host_scheduler_processed_by_block = reset_launch_state()
            timing = PtoRunTiming()
            if (
                runtime.run_prepared(
                    ctx,
                    None,
                    0,
                    ctypes.byref(args),
                    config.block_dim,
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
                raise RuntimeError("run_prepared dag failed")

            if runtime.copy_from_device_ctx(ctx, ctypes.byref(host_tmp0), dev_tmp0, output_nbytes) != 0:
                raise RuntimeError("copy_from_device dag tmp0 failed")
            if runtime.copy_from_device_ctx(ctx, ctypes.byref(host_tmp1), dev_tmp1, output_nbytes) != 0:
                raise RuntimeError("copy_from_device dag tmp1 failed")
            if runtime.copy_from_device_ctx(ctx, ctypes.byref(host_tmp2), dev_tmp2, output_nbytes) != 0:
                raise RuntimeError("copy_from_device dag tmp2 failed")
            if runtime.copy_from_device_ctx(ctx, ctypes.byref(host_tmp3), dev_tmp3, output_nbytes) != 0:
                raise RuntimeError("copy_from_device dag tmp3 failed")
            if runtime.copy_from_device_ctx(ctx, ctypes.byref(host_out), dev_out, output_nbytes) != 0:
                raise RuntimeError("copy_from_device dag out failed")
            if (
                runtime.copy_from_device_ctx(
                    ctx, ctypes.byref(host_counters), dev_counters, ctypes.sizeof(host_counters)
                )
                != 0
            ):
                raise RuntimeError("copy_from_device dag counters failed")
            if runtime.copy_from_device_ctx(ctx, ctypes.byref(host_fanin), dev_fanin, ctypes.sizeof(host_fanin)) != 0:
                raise RuntimeError("copy_from_device dag fanin failed")
            if (
                runtime.copy_from_device_ctx(
                    ctx,
                    ctypes.byref(host_scheduler_processed_by_block),
                    dev_scheduler_processed_by_block,
                    ctypes.sizeof(host_scheduler_processed_by_block),
                )
                != 0
            ):
                raise RuntimeError("copy_from_device dag scheduler processed by block failed")

            error_count = int(host_counters[5])
            error_code = int(host_counters[6])
            error_task_id = int(host_counters[7])
            scheduler_init_count = int(host_counters[8])
            scheduler_loop_count = int(host_counters[9])
            scheduler_processed_count = int(host_counters[10])
            scheduler_processed_by_block = [int(value) for value in host_scheduler_processed_by_block]
            if error_count != 0:
                raise RuntimeError(
                    f"persistent dag scheduler error code={error_code} task_id={error_task_id} count={error_count}"
                )

            expected_tmp0 = [_f32(host_a[i] + host_b[i]) for i in range(n)]
            expected_tmp1 = [_f32(host_a[i] * host_b[i]) for i in range(n)]
            expected_tmp2 = [_f32(expected_tmp0[i] + expected_tmp1[i]) for i in range(n)]
            expected_tmp3 = [_f32(expected_tmp2[i] * host_b[i]) for i in range(n)]
            expected_out = expected_tmp2
            if config.dag_shape in {"chain", "graph_descriptor_chain", "normal_graph_chain"}:
                expected_out = [_f32(expected_tmp2[i] + expected_tmp3[i]) for i in range(n)]
            if config.dag_shape in {"scratch_reuse", "graph_descriptor_scratch_reuse"}:
                expected_tmp0 = [_f32(expected_tmp2[i] + host_a[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp0[i] + expected_tmp3[i]) for i in range(n)]
            if config.dag_shape == "graph_descriptor_parallel_chains":
                root_add = [_f32(host_a[i] + host_b[i]) for i in range(n)]
                root_mul = [_f32(host_a[i] * host_b[i]) for i in range(n)]
                expected_tmp0 = [_f32(root_add[i] + root_mul[i]) for i in range(n)]
                expected_tmp2 = [_f32(root_add[i] + root_mul[i]) for i in range(n)]
                expected_tmp1 = [_f32(expected_tmp0[i] * expected_tmp2[i]) for i in range(n)]
                expected_tmp3 = [_f32(expected_tmp0[i] + expected_tmp2[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp3[i]) for i in range(n)]
            if config.dag_shape == "graph_descriptor_wide_fanout":
                root = [_f32(host_a[i] + host_b[i]) for i in range(n)]
                child_add_a = [_f32(root[i] + host_a[i]) for i in range(n)]
                child_mul_b = [_f32(root[i] * host_b[i]) for i in range(n)]
                expected_tmp3 = [_f32(root[i] + host_b[i]) for i in range(n)]
                expected_tmp1 = [_f32(child_add_a[i] + child_mul_b[i]) for i in range(n)]
                expected_tmp2 = child_mul_b
                expected_tmp0 = [_f32(child_mul_b[i] * expected_tmp3[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp0[i]) for i in range(n)]
            if config.dag_shape in {"graph_descriptor_multi_fanin", "normal_graph_multi_fanin"}:
                expected_tmp0 = [_f32(host_a[i] + host_b[i]) for i in range(n)]
                expected_tmp1 = [_f32(host_a[i] * host_b[i]) for i in range(n)]
                expected_tmp2 = [_f32(2.0 * host_a[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp0[i] * expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
            if config.dag_shape in {"graph_descriptor_layered_cross", "normal_graph_layered_cross"}:
                root_add = [_f32(host_a[i] + host_b[i]) for i in range(n)]
                root_mul = [_f32(host_a[i] * host_b[i]) for i in range(n)]
                root_scale = [_f32(2.0 * host_a[i]) for i in range(n)]
                expected_tmp3 = [_f32(root_add[i] + root_mul[i]) for i in range(n)]
                expected_tmp0 = [_f32(root_mul[i] * root_scale[i]) for i in range(n)]
                side_branch = [_f32(root_scale[i] + host_a[i]) for i in range(n)]
                expected_tmp1 = [_fma_f32(expected_tmp3[i], expected_tmp0[i], host_a[i]) for i in range(n)]
                expected_tmp2 = [_f32(expected_tmp3[i] + side_branch[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
            if config.dag_shape in {"scalar_axpy", "graph_descriptor_scalar_axpy"}:
                expected_tmp0 = [_f32(_f32(1.5 * host_a[i]) + host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp0[i] + expected_tmp1[i]) for i in range(n)]
            if config.dag_shape in {"scalar_scale", "graph_descriptor_scalar_scale"}:
                expected_tmp0 = [_f32(2.0 * host_a[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp0[i] + expected_tmp1[i]) for i in range(n)]
            if config.dag_shape in {"scalar_affine", "graph_descriptor_scalar_affine"}:
                expected_tmp0 = [_f32(_f32(1.5 * host_a[i]) + _f32(0.5 * host_b[i])) for i in range(n)]
                expected_out = [_f32(expected_tmp0[i] + expected_tmp1[i]) for i in range(n)]
            if config.dag_shape in triad_shapes:
                expected_tmp0 = [_f32(3 * i) for i in range(n)]
                expected_tmp1 = [_fma_f32(host_a[i], host_b[i], expected_tmp0[i]) for i in range(n)]
                expected_tmp2 = [_f32(host_a[i] * host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
            if config.dag_shape in quad_shapes:
                expected_tmp0 = [_f32(3 * i) for i in range(n)]
                expected_tmp3 = [_f32(4 * i) for i in range(n)]
                expected_tmp1 = [
                    _fma_f32(host_a[i], host_b[i], _f32(expected_tmp0[i] * expected_tmp3[i])) for i in range(n)
                ]
                expected_tmp2 = [_f32(host_a[i] * host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
            if config.dag_shape in {
                CPP_ORCHESTRATOR_SNAPSHOT_DAG_SHAPE,
                "graph_descriptor_compact_role_inout",
                "graph_descriptor_pair_inout",
                "graph_descriptor_role_map_inout",
                "graph_descriptor_tagged_inout",
                "graph_descriptor_role_keyed_inout",
                "graph_descriptor_submits",
            }:
                expected_tmp0 = [0.0 for _ in range(n)]
                expected_tmp1 = [_f32(_f32(host_a[i] + host_b[i]) + host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + host_a[i]) for i in range(n)]
            if config.dag_shape == "graph_descriptor_submit_groups":
                expected_tmp0 = [0.0 for _ in range(n)]
                expected_tmp1 = [_f32(host_a[i] + host_b[i]) for i in range(n)]
                expected_tmp2 = [_f32(host_a[i] + host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
            if config.dag_shape in graph_arg_shapes:
                expected_tmp0 = [_f32(3 * i) for i in range(n)]
                expected_tmp3 = [_f32(4 * i) for i in range(n)]
                expected_tmp1 = [
                    _f32(_f32(1.5 * host_a[i]) + _f32(expected_tmp0[i] + _f32(0.25 * expected_tmp3[i])))
                    for i in range(n)
                ]
                if config.dag_shape in {"generic_args4", "graph_descriptor_generic_args4"}:
                    expected_tmp1 = [
                        _f32(expected_tmp1[i] + _f32(0.125 * host_a[i]) + _f32(0.0625 * host_b[i])) for i in range(n)
                    ]
                expected_tmp2 = [_f32(host_a[i] * host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
                if config.dag_shape == "graph_descriptor_diamond":
                    expected_tmp0 = [_f32(expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
                    expected_tmp3 = [_f32(expected_tmp1[i] * expected_tmp2[i]) for i in range(n)]
                    expected_out = [_f32(expected_tmp0[i] + expected_tmp3[i]) for i in range(n)]
            if config.dag_shape in {"unary_square", "graph_descriptor_unary_square"}:
                expected_tmp0 = [_f32(host_a[i] * host_a[i]) for i in range(n)]
                expected_tmp1 = [_f32(expected_tmp0[i] + host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + host_a[i]) for i in range(n)]
            if config.dag_shape in {
                "graph_descriptor_depends_on",
                "graph_descriptor_node_io",
                "graph_descriptor_node_link",
                "graph_descriptor_named_callable",
                "graph_descriptor_node_op",
                "graph_descriptor_node_port_dict",
                "graph_descriptor_task_dict",
            }:
                expected_out = expected_tmp0
            if _is_tensor_tile_shape(config.dag_shape):
                expected_tmp0 = _matmul_expected(host_a, host_b, n, config.tensor_tile)
                expected_tmp1 = [_f32(expected_tmp0[i] + host_a[i]) for i in range(n)]
                expected_tmp2 = [_f32(expected_tmp0[i] * host_b[i]) for i in range(n)]
                expected_out = [_f32(expected_tmp1[i] + expected_tmp2[i]) for i in range(n)]
            if list(host_tmp0) != expected_tmp0:
                mismatch = _first_mismatch(list(host_tmp0), expected_tmp0)
                raise RuntimeError(f"dag tmp0 mismatch on launch {launch_idx}: {mismatch}")
            tmp1_mismatch = _first_mismatch(list(host_tmp1), expected_tmp1, rtol=1e-6, atol=1e-5)
            if tmp1_mismatch != "no mismatch":
                mismatch = tmp1_mismatch
                detail = ""
                if mismatch.startswith("idx="):
                    idx = int(mismatch.split(" ", 1)[0].split("=", 1)[1])
                    detail = (
                        f" tmp0={float(host_tmp0[idx])} expected_tmp0={float(expected_tmp0[idx])}"
                        f" tmp2={float(host_tmp2[idx])} expected_tmp2={float(expected_tmp2[idx])}"
                        f" tmp3={float(host_tmp3[idx])} expected_tmp3={float(expected_tmp3[idx])}"
                        f" out={float(host_out[idx])} expected_out={float(expected_out[idx])}"
                    )
                raise RuntimeError(f"dag tmp1 mismatch on launch {launch_idx}: {mismatch}{detail}")
            if (
                config.dag_shape
                in {
                    "chain",
                    "graph_descriptor_chain",
                    "normal_graph_chain",
                    "scratch_reuse",
                    "graph_tensor_tile",
                    "tensor_core_tile",
                    "tensor_tile",
                    "triad",
                    "graph_descriptor_triad",
                    "quad",
                    "graph_descriptor_quad",
                    "generic_args",
                    "generic_args4",
                    "graph_descriptor",
                    "graph_descriptor_diamond",
                    "graph_descriptor_layered_cross",
                    "normal_graph_layered_cross",
                    "graph_descriptor_generic_args4",
                    "graph_descriptor_multi_fanin",
                    "normal_graph_multi_fanin",
                    "graph_descriptor_node_attrs",
                    "graph_descriptor_parallel_chains",
                    "graph_descriptor_reordered",
                    "graph_descriptor_scratch_reuse",
                    "graph_descriptor_submit_groups",
                    "graph_descriptor_tagged",
                    "graph_descriptor_wide_fanout",
                }
                and list(host_tmp2) != expected_tmp2
            ):
                raise RuntimeError(f"dag tmp2 mismatch on launch {launch_idx}")
            if (
                config.dag_shape
                in {
                    "chain",
                    "graph_descriptor_chain",
                    "normal_graph_chain",
                    "graph_descriptor_layered_cross",
                    "normal_graph_layered_cross",
                    "graph_descriptor_parallel_chains",
                    "graph_descriptor_wide_fanout",
                    "scratch_reuse",
                    "graph_descriptor_scratch_reuse",
                    "quad",
                    "graph_descriptor_quad",
                    *graph_arg_shapes,
                }
                and list(host_tmp3) != expected_tmp3
            ):
                raise RuntimeError(f"dag tmp3 mismatch on launch {launch_idx}")
            output_mismatch = _first_mismatch(list(host_out), expected_out, rtol=1e-6, atol=1e-5)
            if output_mismatch != "no mismatch":
                raise RuntimeError(f"dag output mismatch on launch {launch_idx}: {output_mismatch}")
            completed_count = int(host_counters[4])
            if completed_count != task_count:
                raise RuntimeError(f"persistent dag completed {completed_count}/{task_count} tasks")
            launch_completed_counts.append(completed_count)
            launch_host_wall_ns.append(timing.host_wall_ns)
            launch_device_wall_ns.append(timing.device_wall_ns)
        if runtime.unregister_callable(ctx, 0) != 0:
            raise RuntimeError("unregister_callable dag failed")

        result = {
            "status": "pass",
            "runtime": "persistent_device",
            "mode": "dag",
            "dag_shape": config.dag_shape,
            "device": config.device,
            "task_count": task_count,
            "n": n,
            "queue_capacity": queue_capacity,
            "scheduler_blocks": scheduler_blocks,
            "worker_blocks": worker_blocks,
            "worker_blocks_per_task": 1,
            "stream_id": config.stream_id,
            "repeat_runs": config.repeat_runs,
            "resource_policy": {
                "scheduler_blocks": scheduler_blocks,
                "worker_blocks": worker_blocks,
                "worker_blocks_per_task": 1,
                "stream_id": config.stream_id,
                "block_dim": config.block_dim,
                "grid_dim": scheduler_blocks + worker_blocks,
            },
            "completed_count": completed_count,
            "scheduler_init_count": scheduler_init_count,
            "scheduler_loop_count": scheduler_loop_count,
            "scheduler_processed_count": scheduler_processed_count,
            "scheduler_processed_by_block": scheduler_processed_by_block,
            "launch_completed_counts": launch_completed_counts,
            "device_scheduler_errors": {
                "count": error_count,
                "code": error_code,
                "task_id": error_task_id,
            },
            "dispatch_func_ids": [int(task.func_id) for task in tasks],
            "fanin_remaining": [int(value) for value in host_fanin],
            "ptx_arch": config.arch,
            "ptx_source": config.ptx_source,
            "source_kind": "generated-dispatch",
            "host_wall_ns": sum(launch_host_wall_ns),
            "device_wall_ns": sum(launch_device_wall_ns),
            "launch_host_wall_ns": launch_host_wall_ns,
            "launch_device_wall_ns": launch_device_wall_ns,
            "elapsed_s": time.time() - config.start,
            "host_runtime": str(config.binaries.host_path),
        }
        if config.dag_shape in {"scratch_reuse", "graph_descriptor_scratch_reuse"}:
            result["scratch_reuse"] = {"reused_buffer": "tmp0", "reuse_task": 4}
        if _is_tensor_tile_shape(config.dag_shape):
            result["tensor_tile"] = {
                **config.tensor_tile,
                "tile_count": (tensor_lengths or _tensor_tile_buffer_lengths(n, config.tensor_tile))["tile_count"],
            }
        if config.dag_shape in {"graph_tensor_core_tile", "tensor_core_tile"}:
            result["tensor_core"] = {"api": "wmma", "mma_shape": "m16n16k8", "input": "tf32", "accumulator": "f32"}
        if config.dag_shape in {"scalar_axpy", "graph_descriptor_scalar_axpy"}:
            result["scalar_args"] = {"scalar0": 1.5}
        if config.dag_shape in {"scalar_scale", "graph_descriptor_scalar_scale"}:
            result["scalar_args"] = {"scalar0": 2.0}
        if config.dag_shape in {"scalar_affine", "graph_descriptor_scalar_affine"}:
            result["scalar_args"] = {"scalar0": 1.5, "scalar1": 0.5}
        if config.dag_shape in {"graph_descriptor_multi_fanin", "normal_graph_multi_fanin"}:
            result["scalar_args"] = {"scalar0": 2.0}
            result["tensor_args"] = {"c": "tmp2"}
        if config.dag_shape in {"graph_descriptor_layered_cross", "normal_graph_layered_cross"}:
            result["scalar_args"] = {"scalar0": 2.0}
            result["tensor_args"] = {"c": "a"}
        if config.dag_shape in triad_shapes:
            result["tensor_args"] = {"c": "tmp0"}
        if config.dag_shape in quad_shapes:
            result["tensor_args"] = {"c": "tmp0", "d": "tmp3"}
        if config.dag_shape in {"generic_args", "generic_args4"}:
            tensor_args = {"0": "tmp0", "1": "tmp3"}
            scalar_args = [1.5, 0.25]
            if config.dag_shape == "generic_args4":
                tensor_args.update({"2": "a", "3": "b"})
                scalar_args.extend([0.125, 0.0625])
            result["generic_args"] = {
                "tensor_args": tensor_args,
                "scalar_args": scalar_args,
            }
            result["tensor_args"] = {f"tensor_args[{idx}]": value for idx, value in tensor_args.items()}
            result["scalar_args"] = {f"scalar_args[{idx}]": value for idx, value in enumerate(scalar_args)}
        if config.dag_shape in {
            "graph_descriptor",
            "graph_descriptor_scalar_affine",
            "graph_descriptor_scalar_axpy",
            "graph_descriptor_chain",
            "graph_descriptor_compact_role_inout",
            "graph_descriptor_depends_on",
            "graph_descriptor_diamond",
            "graph_descriptor_generic_args4",
            "graph_descriptor_layered_cross",
            "graph_descriptor_multi_fanin",
            "graph_descriptor_node_attrs",
            "graph_descriptor_node_io",
            "graph_descriptor_node_link",
            "graph_descriptor_named_callable",
            "graph_descriptor_node_op",
            "graph_descriptor_node_port_dict",
            "graph_descriptor_pair_inout",
            "graph_descriptor_parallel_chains",
            "graph_descriptor_quad",
            "graph_descriptor_reordered",
            "graph_descriptor_role_map_inout",
            "graph_descriptor_role_keyed_inout",
            "graph_descriptor_scalar_scale",
            "graph_descriptor_scratch_reuse",
            "graph_descriptor_submit_groups",
            "graph_descriptor_submits",
            "graph_descriptor_task_dict",
            "graph_descriptor_tagged",
            "graph_descriptor_tagged_inout",
            "graph_descriptor_triad",
            "graph_descriptor_unary_square",
            "graph_descriptor_wide_fanout",
            "graph_tensor_core_tile",
            "graph_tensor_tile",
            *NORMAL_GRAPH_DAG_SHAPES,
        }:
            result["graph_descriptor"] = {
                "tasks": task_count,
                "dependents": [int(value) for value in dependents],
                "fanin": initial_fanin,
            }
            if config.dag_shape in {
                "graph_descriptor",
                "graph_descriptor_diamond",
                "graph_descriptor_generic_args4",
                "graph_descriptor_node_attrs",
                "graph_descriptor_reordered",
                "graph_descriptor_tagged",
            }:
                tensor_args = {"tensor_args[0]": "tmp0", "tensor_args[1]": "tmp3"}
                scalar_args = {"scalar_args[0]": 1.5, "scalar_args[1]": 0.25}
                if config.dag_shape == "graph_descriptor_generic_args4":
                    tensor_args.update({"tensor_args[2]": "a", "tensor_args[3]": "b"})
                    scalar_args.update({"scalar_args[2]": 0.125, "scalar_args[3]": 0.0625})
                result["tensor_args"] = tensor_args
                result["scalar_args"] = scalar_args
            if config.dag_shape == "graph_descriptor_node_attrs":
                result["graph_node_attrs"] = {"task0": "attrs:tensor_args,scalar_args"}
            if config.dag_shape in {
                "graph_descriptor_node_link",
                "graph_descriptor_named_callable",
                "graph_descriptor_node_op",
                "graph_descriptor_node_port_dict",
            }:
                result["graph_node_ops"] = {
                    "task0": "op:add=1",
                    "task1": "op:mul=2",
                    "task2": "op:add=1",
                }
            if config.dag_shape == "graph_descriptor_node_port_dict":
                result["graph_task_arg_key"] = "node_port_dict"
                result["graph_task_args"] = {
                    "task0": "input.lhs:a,input.rhs:b,output.value:tmp0",
                    "task1": "input.lhs:a,input.rhs:b,output.value:tmp1",
                    "task2": "input.lhs:tmp0,input.rhs:tmp1,output.value:out",
                }
            if config.dag_shape == "graph_descriptor_named_callable":
                result["graph_task_arg_key"] = "named_callable"
                result["graph_task_args"] = {
                    "task0": "callable:add,input:a,input:b,output:tmp0",
                    "task1": "callable:mul,input:a,input:b,output:tmp1",
                    "task2": "callable:add,input:a,input:b,output:out",
                }
            if config.dag_shape == "graph_descriptor_node_io":
                result["graph_task_arg_key"] = "node_io"
                result["graph_task_args"] = {
                    "task0": "input:a,input:b,output:tmp0",
                    "task1": "input:a,input:b,output:tmp1",
                    "task2": "input:a,input:b,output:out",
                }
            if config.dag_shape == "graph_descriptor_task_dict":
                result["graph_task_arg_key"] = "task_dict"
                result["graph_task_args"] = {
                    "left": "input:a,input:b,output:tmp0",
                    "right": "input:a,input:b,output:tmp1",
                    "join": "input:a,input:b,output:out",
                }
            if config.dag_shape == "graph_descriptor_tagged":
                result["graph_task_args"] = {
                    "task0": "input:a,input:b,output:tmp1,scalar:scalar_args[0],scalar:scalar_args[1]",
                    "task1": "input:a,input:b,output:tmp2",
                    "task2": "input:tmp1,input:tmp2,output_existing:out",
                }
            if config.dag_shape in {
                "graph_descriptor_compact_role_inout",
                "graph_descriptor_pair_inout",
                "graph_descriptor_role_map_inout",
                "graph_descriptor_tagged_inout",
                "graph_descriptor_role_keyed_inout",
                "graph_descriptor_submits",
                *NORMAL_GRAPH_DAG_SHAPES,
            }:
                if config.dag_shape == CPP_ORCHESTRATOR_SNAPSHOT_DAG_SHAPE:
                    result["graph_task_arg_key"] = "cpp_orchestrator_snapshot"
                    result["graph_lowering"] = "normal_graph"
                    result["graph_source"] = "cpp_orchestrator_snapshot"
                    result["graph_task_args"] = {
                        "slot0": "input:a,input:b,output:tmp1",
                        "slot1": "inout:tmp1,input:b",
                        "slot2": "input:tmp1,input:a,output_existing:out",
                    }
                elif config.dag_shape in NORMAL_GRAPH_DAG_SHAPES:
                    result["graph_task_arg_key"] = "normal_graph"
                    result["graph_lowering"] = "normal_graph"
                else:
                    result["graph_task_arg_key"] = {
                        "graph_descriptor_compact_role_inout": "compact",
                        "graph_descriptor_pair_inout": "pair",
                        "graph_descriptor_role_map_inout": "role_map",
                        "graph_descriptor_role_keyed_inout": "role",
                        "graph_descriptor_submits": "submits",
                        "graph_descriptor_tagged_inout": "tag",
                    }[config.dag_shape]
                    result["graph_task_args"] = {
                        "task0": "input:a,input:b,output:tmp1",
                        "task1": "inout:tmp1,input:b",
                        "task2": "input:tmp1,input:a,output_existing:out",
                    }
                    if config.dag_shape == "graph_descriptor_submits":
                        result["graph_lowering"] = "normal_graph"
            if config.dag_shape == "graph_descriptor_submit_groups":
                result["graph_task_arg_key"] = "submit_groups"
                result["graph_task_args"] = {
                    "task0": "input:a,input:b,output:tmp1",
                    "task1": "input:a,input:b,output:tmp2",
                    "task2": "input:tmp1,input:tmp2,output_existing:out",
                }
        return result
    finally:
        for ptr in allocated:
            runtime.device_free_ctx(ctx, ptr)


def _completed_count(runtime, ctx: int, launch: PersistentLaunch, task_count: int) -> int:
    if launch.host_counters is None or launch.dev_counters is None:
        return launch.completed_count
    if (
        runtime.copy_from_device_ctx(
            ctx, ctypes.byref(launch.host_counters), launch.dev_counters, ctypes.sizeof(launch.host_counters)
        )
        != 0
    ):
        raise RuntimeError("copy_from_device queue counters failed")
    completed_count = int(launch.host_counters[2])
    if completed_count != task_count:
        raise RuntimeError(f"persistent queue completed {completed_count}/{task_count} tasks")
    return completed_count


def _reset_queue_launch_state(runtime, ctx: int, launch: PersistentLaunch, queue_capacity: int) -> None:
    if launch.host_counters is None or launch.dev_counters is None:
        return
    counters_t = type(launch.host_counters)
    host_counters = counters_t(*([0] * len(launch.host_counters)))
    launch.host_counters = host_counters
    dev_counters = launch.dev_counters
    if runtime.copy_to_device_ctx(ctx, dev_counters, ctypes.byref(host_counters), ctypes.sizeof(host_counters)) != 0:
        raise RuntimeError("copy_to_device queue counters failed")
    if launch.host_ready_flags is None or launch.dev_ready_flags is None:
        return
    flags_t = type(launch.host_ready_flags)
    host_ready_flags = flags_t(*([0] * queue_capacity))
    launch.host_ready_flags = host_ready_flags
    dev_ready_flags = launch.dev_ready_flags
    if (
        runtime.copy_to_device_ctx(
            ctx,
            dev_ready_flags,
            ctypes.byref(host_ready_flags),
            ctypes.sizeof(host_ready_flags),
        )
        != 0
    ):
        raise RuntimeError("copy_to_device queue flags failed")


def run_persistent_smoke(  # noqa: PLR0912, PLR0913, PLR0915
    device: int,
    task_count: int,
    n: int,
    arch: str,
    mode: str = "direct",
    queue_capacity: int | None = None,
    worker_blocks_per_task: int = 1,
    worker_blocks: int | None = None,
    scheduler_blocks: int = 1,
    stream_id: int = 0,
    block_dim: int = 256,
    dag_shape: str = "fork_join",
    tensor_rows: int = 16,
    tensor_cols: int = 16,
    tensor_inner: int = 16,
    repeat_runs: int = 1,
) -> dict:
    if mode not in {"dag", "direct", "queue"}:
        raise ValueError(f"unknown persistent mode: {mode}")
    if dag_shape not in {
        "bad_dependent",
        "bad_dependent_range",
        "bad_duplicate_dependent",
        "bad_fanin_underflow",
        "bad_func_id",
        "bad_initial_fanin",
        "bad_no_root",
        "bad_self_dependent",
        "bad_unreachable",
        "chain",
        "fork_join",
        "generic_args",
        "generic_args4",
        "graph_descriptor",
        "graph_descriptor_chain",
        "graph_descriptor_compact_role_inout",
        "graph_descriptor_depends_on",
        "graph_descriptor_diamond",
        "graph_descriptor_generic_args4",
        "graph_descriptor_layered_cross",
        "graph_descriptor_multi_fanin",
        "graph_descriptor_node_attrs",
        "graph_descriptor_node_io",
        "graph_descriptor_node_link",
        "graph_descriptor_named_callable",
        "graph_descriptor_node_op",
        "graph_descriptor_node_port_dict",
        "graph_descriptor_pair_inout",
        "graph_descriptor_parallel_chains",
        "graph_descriptor_quad",
        "graph_descriptor_reordered",
        "graph_descriptor_role_map_inout",
        "graph_descriptor_scalar_affine",
        "graph_descriptor_scalar_axpy",
        "graph_descriptor_scalar_scale",
        "graph_descriptor_scratch_reuse",
        "graph_descriptor_role_keyed_inout",
        "graph_descriptor_submit_groups",
        "graph_descriptor_submits",
        "graph_descriptor_task_dict",
        "graph_descriptor_tagged",
        "graph_descriptor_tagged_inout",
        "graph_descriptor_triad",
        "graph_descriptor_unary_square",
        "graph_descriptor_wide_fanout",
        "graph_tensor_core_tile",
        "graph_tensor_tile",
        *NORMAL_GRAPH_DAG_SHAPES,
        "quad",
        "scalar_affine",
        "scalar_axpy",
        "scalar_scale",
        "scratch_reuse",
        "tensor_core_tile",
        "tensor_tile",
        "triad",
        "unary_square",
    }:
        raise ValueError(f"unknown dag shape: {dag_shape}")
    if worker_blocks_per_task <= 0:
        raise ValueError("worker_blocks_per_task must be positive")
    if mode != "direct" and worker_blocks_per_task != 1:
        raise ValueError("worker_blocks_per_task is only supported for direct mode")
    if worker_blocks is not None and worker_blocks <= 0:
        raise ValueError("worker_blocks must be positive")
    if mode == "direct" and worker_blocks is not None:
        raise ValueError("worker_blocks is only supported for queue and dag modes")
    if scheduler_blocks <= 0:
        raise ValueError("scheduler_blocks must be positive")
    if mode == "direct" and scheduler_blocks != 1:
        raise ValueError("scheduler_blocks is only supported for queue and dag modes")
    if stream_id < 0:
        raise ValueError("stream_id must be non-negative")
    if block_dim <= 0:
        raise ValueError("block_dim must be positive")
    if repeat_runs <= 0:
        raise ValueError("repeat_runs must be positive")
    tensor_tile = _make_tensor_tile_descriptor(rows=tensor_rows, cols=tensor_cols, inner=tensor_inner)
    if queue_capacity is None:
        queue_capacity = task_count
    if queue_capacity <= 0:
        raise ValueError("queue_capacity must be positive")
    with tempfile.TemporaryDirectory(prefix="pto_cuda_persistent_") as td:
        ptx, ptx_source, persistent_artifact = _compile_persistent_ptx(Path(td), arch, mode)
    if mode == "direct" and worker_blocks_per_task > 1 and ptx_source.startswith("embedded-"):
        raise RuntimeError("worker_blocks_per_task > 1 requires nvcc-built persistent direct PTX")
    if mode == "queue" and scheduler_blocks > 1 and ptx_source.startswith("embedded-"):
        raise RuntimeError("scheduler_blocks > 1 requires nvcc-built persistent queue PTX")
    if mode == "dag" and _is_tensor_tile_shape(dag_shape) and ptx_source.startswith("embedded-"):
        raise RuntimeError(f"{dag_shape} DAG shape requires nvcc-built generated-dispatch PTX")
    if mode == "dag" and dag_shape in {"triad", "graph_descriptor_triad"} and ptx_source.startswith("embedded-"):
        raise RuntimeError(f"{dag_shape} DAG shape requires nvcc-built generated-dispatch PTX")
    if mode == "dag" and dag_shape in {"quad", "graph_descriptor_quad"} and ptx_source.startswith("embedded-"):
        raise RuntimeError(f"{dag_shape} DAG shape requires nvcc-built generated-dispatch PTX")
    if (
        mode == "dag"
        and dag_shape
        in {
            "generic_args",
            "generic_args4",
            "graph_descriptor",
            "graph_descriptor_chain",
            "graph_descriptor_compact_role_inout",
            "graph_descriptor_depends_on",
            "graph_descriptor_diamond",
            "graph_descriptor_generic_args4",
            "graph_descriptor_layered_cross",
            "graph_descriptor_multi_fanin",
            "graph_descriptor_node_attrs",
            "graph_descriptor_node_io",
            "graph_descriptor_node_link",
            "graph_descriptor_named_callable",
            "graph_descriptor_node_op",
            "graph_descriptor_node_port_dict",
            "graph_descriptor_pair_inout",
            "graph_descriptor_parallel_chains",
            "graph_descriptor_reordered",
            "graph_descriptor_role_map_inout",
            "graph_descriptor_scalar_affine",
            "graph_descriptor_scalar_axpy",
            "graph_descriptor_scalar_scale",
            "graph_descriptor_submit_groups",
            "graph_descriptor_submits",
            "graph_descriptor_role_keyed_inout",
            "graph_descriptor_task_dict",
            "graph_descriptor_tagged",
            "graph_descriptor_tagged_inout",
            "graph_descriptor_wide_fanout",
            *NORMAL_GRAPH_DAG_SHAPES,
        }
        and ptx_source.startswith("embedded-")
    ):
        raise RuntimeError(f"{dag_shape} DAG shape requires nvcc-built generated-dispatch PTX")
    if (
        mode == "dag"
        and dag_shape in {"unary_square", "graph_descriptor_unary_square"}
        and ptx_source.startswith("embedded-")
    ):
        raise RuntimeError(f"{dag_shape} DAG shape requires nvcc-built generated-dispatch PTX")
    if (
        mode == "dag"
        and dag_shape in {"scalar_scale", "graph_descriptor_scalar_scale"}
        and ptx_source.startswith("embedded-")
    ):
        raise RuntimeError(f"{dag_shape} DAG shape requires nvcc-built generated-dispatch PTX")
    ptx_buf = ctypes.create_string_buffer(ptx + b"\0")

    runtime, binaries = _load_persistent_runtime()
    ctx = runtime.create_device_context()
    if not ctx:
        raise RuntimeError("create_device_context returned null")

    start = time.time()
    try:
        if runtime.simpler_init(ctx, device, None, 0, None, 0) != 0:
            raise RuntimeError("simpler_init failed")
        if mode == "dag":
            return _run_dag_smoke(
                DagSmokeConfig(
                    runtime=runtime,
                    binaries=binaries,
                    ctx=ctx,
                    device=device,
                    n=n,
                    arch=arch,
                    queue_capacity=queue_capacity,
                    scheduler_blocks=scheduler_blocks,
                    worker_blocks=worker_blocks if worker_blocks is not None else task_count,
                    stream_id=stream_id,
                    block_dim=block_dim,
                    ptx=ptx,
                    ptx_buf=ptx_buf,
                    persistent_artifact=persistent_artifact,
                    ptx_source=ptx_source,
                    start=start,
                    dag_shape=dag_shape,
                    tensor_tile=tensor_tile,
                    repeat_runs=repeat_runs,
                )
            )

        array_t = ctypes.c_float * n
        host_a = array_t(*[float(i) for i in range(n)])
        host_b = array_t(*[float(2 * i) for i in range(n)])
        host_out = [array_t() for _ in range(task_count)]
        nbytes = ctypes.sizeof(host_a)

        dev_a = runtime.device_malloc_ctx(ctx, nbytes)
        dev_b = runtime.device_malloc_ctx(ctx, nbytes)
        dev_out = [runtime.device_malloc_ctx(ctx, nbytes) for _ in range(task_count)]
        task_array_t = CudaPersistentVectorAddTask * task_count
        task_desc = task_array_t(
            *[CudaPersistentVectorAddTask(a=dev_a, b=dev_b, out=dev_out[idx], n=n) for idx in range(task_count)]
        )
        dev_tasks = runtime.device_malloc_ctx(ctx, ctypes.sizeof(task_desc))
        launch = None
        if not (dev_a and dev_b and all(dev_out) and dev_tasks):
            raise RuntimeError("device allocation failed")
        try:
            if runtime.copy_to_device_ctx(ctx, dev_a, ctypes.byref(host_a), nbytes) != 0:
                raise RuntimeError("copy_to_device a failed")
            if runtime.copy_to_device_ctx(ctx, dev_b, ctypes.byref(host_b), nbytes) != 0:
                raise RuntimeError("copy_to_device b failed")
            if runtime.copy_to_device_ctx(ctx, dev_tasks, ctypes.byref(task_desc), ctypes.sizeof(task_desc)) != 0:
                raise RuntimeError("copy_to_device tasks failed")

            launch = _make_launch(
                runtime,
                ctx,
                mode,
                ptx_buf,
                len(ptx) + 1,
                task_count,
                queue_capacity,
                dev_tasks,
                worker_blocks_per_task,
                worker_blocks,
                scheduler_blocks,
                stream_id,
                block_dim,
            )
            if runtime.prepare_callable(ctx, 0, ctypes.byref(launch.manifest)) != 0:
                raise RuntimeError("prepare_callable failed")
            expected = [float(3 * i) for i in range(n)]
            launch_completed_counts = []
            launch_host_wall_ns = []
            launch_device_wall_ns = []
            completed_count = 0
            for launch_idx in range(repeat_runs):
                _reset_queue_launch_state(runtime, ctx, launch, queue_capacity)
                timing = PtoRunTiming()
                if (
                    runtime.run_prepared(
                        ctx,
                        None,
                        0,
                        ctypes.byref(launch.args),
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
                    raise RuntimeError("run_prepared failed")

                for idx, out in enumerate(host_out):
                    if runtime.copy_from_device_ctx(ctx, ctypes.byref(out), dev_out[idx], nbytes) != 0:
                        raise RuntimeError(f"copy_from_device {idx} failed")
                    if list(out) != expected:
                        raise RuntimeError(f"persistent output {idx} mismatch on launch {launch_idx}")
                completed_count = _completed_count(runtime, ctx, launch, task_count)
                launch_completed_counts.append(completed_count)
                launch_host_wall_ns.append(timing.host_wall_ns)
                launch_device_wall_ns.append(timing.device_wall_ns)
            if runtime.unregister_callable(ctx, 0) != 0:
                raise RuntimeError("unregister_callable failed")
        finally:
            runtime.device_free_ctx(ctx, dev_a)
            runtime.device_free_ctx(ctx, dev_b)
            for ptr in dev_out:
                runtime.device_free_ctx(ctx, ptr)
            runtime.device_free_ctx(ctx, dev_tasks)
            if launch is not None:
                for ptr in launch.device_allocations:
                    runtime.device_free_ctx(ctx, ptr)
    finally:
        runtime.finalize_device(ctx)
        runtime.destroy_device_context(ctx)

    if launch is None:
        raise RuntimeError("persistent launch was not created")

    return {
        "status": "pass",
        "runtime": "persistent_device",
        "mode": mode,
        "device": device,
        "task_count": task_count,
        "n": n,
        "queue_capacity": queue_capacity,
        "scheduler_blocks": launch.scheduler_blocks,
        "worker_blocks": launch.worker_blocks,
        "worker_blocks_per_task": worker_blocks_per_task,
        "stream_id": stream_id,
        "repeat_runs": repeat_runs,
        "resource_policy": {
            "scheduler_blocks": launch.scheduler_blocks,
            "worker_blocks": launch.worker_blocks,
            "worker_blocks_per_task": worker_blocks_per_task,
            "stream_id": stream_id,
            "block_dim": block_dim,
            "grid_dim": launch.scheduler_blocks + launch.worker_blocks,
        },
        "completed_count": completed_count,
        "launch_completed_counts": launch_completed_counts,
        "ptx_arch": arch,
        "ptx_source": ptx_source,
        "host_wall_ns": sum(launch_host_wall_ns),
        "device_wall_ns": sum(launch_device_wall_ns),
        "launch_host_wall_ns": launch_host_wall_ns,
        "launch_device_wall_ns": launch_device_wall_ns,
        "elapsed_s": time.time() - start,
        "host_runtime": str(binaries.host_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=2)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--arch", default="compute_80")
    parser.add_argument("--mode", choices=["dag", "direct", "queue"], default="direct")
    parser.add_argument("--queue-capacity", type=int, default=None)
    parser.add_argument("--worker-blocks-per-task", type=int, default=1)
    parser.add_argument("--worker-blocks", type=int, default=None)
    parser.add_argument("--scheduler-blocks", type=int, default=1)
    parser.add_argument("--stream-id", type=int, default=0)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument(
        "--dag-shape",
        choices=[
            "bad_dependent",
            "bad_dependent_range",
            "bad_duplicate_dependent",
            "bad_fanin_underflow",
            "bad_func_id",
            "bad_initial_fanin",
            "bad_no_root",
            "bad_self_dependent",
            "bad_unreachable",
            "chain",
            "fork_join",
            "generic_args",
            "generic_args4",
            "graph_descriptor",
            "graph_descriptor_chain",
            "graph_descriptor_compact_role_inout",
            "graph_descriptor_depends_on",
            "graph_descriptor_diamond",
            "graph_descriptor_generic_args4",
            "graph_descriptor_layered_cross",
            "graph_descriptor_multi_fanin",
            "graph_descriptor_node_attrs",
            "graph_descriptor_node_io",
            "graph_descriptor_node_link",
            "graph_descriptor_named_callable",
            "graph_descriptor_node_op",
            "graph_descriptor_node_port_dict",
            "graph_descriptor_pair_inout",
            "graph_descriptor_parallel_chains",
            "graph_descriptor_quad",
            "graph_descriptor_reordered",
            "graph_descriptor_role_map_inout",
            "graph_descriptor_scalar_affine",
            "graph_descriptor_scalar_axpy",
            "graph_descriptor_scalar_scale",
            "graph_descriptor_scratch_reuse",
            "graph_descriptor_role_keyed_inout",
            "graph_descriptor_submit_groups",
            "graph_descriptor_submits",
            "graph_descriptor_task_dict",
            "graph_descriptor_tagged",
            "graph_descriptor_tagged_inout",
            "graph_descriptor_triad",
            "graph_descriptor_unary_square",
            "graph_descriptor_wide_fanout",
            "graph_tensor_core_tile",
            "graph_tensor_tile",
            *NORMAL_GRAPH_DAG_SHAPES,
            "quad",
            "scalar_affine",
            "scalar_axpy",
            "scalar_scale",
            "scratch_reuse",
            "tensor_core_tile",
            "tensor_tile",
            "triad",
            "unary_square",
        ],
        default="fork_join",
    )
    parser.add_argument("--tensor-rows", type=int, default=16)
    parser.add_argument("--tensor-cols", type=int, default=16)
    parser.add_argument("--tensor-inner", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    rendered = json.dumps(
        run_persistent_smoke(
            device=args.device,
            task_count=args.task_count,
            n=args.n,
            arch=args.arch,
            mode=args.mode,
            queue_capacity=args.queue_capacity,
            worker_blocks_per_task=args.worker_blocks_per_task,
            worker_blocks=args.worker_blocks,
            scheduler_blocks=args.scheduler_blocks,
            stream_id=args.stream_id,
            block_dim=args.block_dim,
            dag_shape=args.dag_shape,
            tensor_rows=args.tensor_rows,
            tensor_cols=args.tensor_cols,
            tensor_inner=args.tensor_inner,
            repeat_runs=args.repeat_runs,
        ),
        indent=2,
        sort_keys=True,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
