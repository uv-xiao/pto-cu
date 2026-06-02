"""Persistent DAG descriptors for the controlled Qwen microdecode proxy."""

from __future__ import annotations

import ctypes
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagState,
    CudaPersistentDagTask,
)

from qwen_persistent_microdecode_live_impl.plan import CALLABLES


def make_tasks(ptrs: dict[str, int]) -> Any:
    tensor_args_t = ctypes.c_void_p * 5
    scalar_args_t = ctypes.c_float * 4
    task_t = CudaPersistentDagTask * 3
    zero_scalars = scalar_args_t(0.0, 0.0, 0.0, 0.0)
    return task_t(
        CudaPersistentDagTask(
            func_id=CALLABLES[0][1],
            a=ptrs["a"],
            b=ptrs["b"],
            out=ptrs["qkv"],
            n=4,
            dependent_begin=0,
            dependent_count=1,
            initial_fanin=0,
            c=ptrs["c"],
            d=ptrs["d"],
            tensor_args=tensor_args_t(ptrs["q_weight"], None, None, None, None),
            scalar_args=zero_scalars,
            tensor_arg_count=1,
            scalar_arg_count=0,
        ),
        CudaPersistentDagTask(
            func_id=CALLABLES[1][1],
            a=ptrs["qkv"],
            out=ptrs["attn_o"],
            n=4,
            dependent_begin=1,
            dependent_count=1,
            initial_fanin=1,
            tensor_args=tensor_args_t(ptrs["o_weight"], None, None, None, None),
            scalar_args=zero_scalars,
            tensor_arg_count=1,
            scalar_arg_count=0,
        ),
        CudaPersistentDagTask(
            func_id=CALLABLES[2][1],
            a=ptrs["attn_o"],
            out=ptrs["logits"],
            n=4,
            dependent_begin=0,
            dependent_count=0,
            initial_fanin=1,
            tensor_args=tensor_args_t(ptrs["q_weight"], None, None, None, None),
            scalar_args=zero_scalars,
            tensor_arg_count=1,
            scalar_arg_count=0,
        ),
    )


def make_state(plan: dict[str, Any], ptrs: dict[str, int]) -> CudaPersistentDagState:
    word = ctypes.sizeof(ctypes.c_uint32)
    return CudaPersistentDagState(
        tasks=ptrs["tasks"],
        task_count=3,
        dependents=ptrs["dependents"],
        dependent_count=2,
        fanin=ptrs["fanin"],
        ready_queue=ptrs["ready_queue"],
        ready_flags=ptrs["ready_flags"],
        completion_queue=ptrs["completion_queue"],
        completion_flags=ptrs["completion_flags"],
        queue_capacity=plan["dag"]["queue_capacity"],
        queue_head=ptrs["counters"],
        queue_tail=ptrs["counters"] + word,
        completion_head=ptrs["counters"] + 2 * word,
        completion_tail=ptrs["counters"] + 3 * word,
        completed_count=ptrs["counters"] + 4 * word,
        error_count=ptrs["counters"] + 5 * word,
        error_code=ptrs["counters"] + 6 * word,
        error_task_id=ptrs["counters"] + 7 * word,
        scheduler_blocks=plan["dag"]["scheduler_blocks"],
        scheduler_init_count=ptrs["counters"] + 8 * word,
        scheduler_loop_count=ptrs["counters"] + 9 * word,
        scheduler_processed_count=ptrs["counters"] + 10 * word,
        scheduler_processed_by_block=ptrs["scheduler_processed"],
    )
