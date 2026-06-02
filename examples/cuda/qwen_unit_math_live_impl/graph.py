"""Persistent DAG descriptors for Qwen unit-math live execution."""

from __future__ import annotations

import ctypes
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagState,
    CudaPersistentDagTask,
)
from simpler_setup.cuda_pto_graph import (
    CudaPtoTaskArg,
    CudaPtoTaskSubmit,
    lower_cuda_pto_task_graph,
)

from qwen_unit_math_live_impl.plan import CALLABLES


def make_graph_arrays(ptrs: dict[str, int]) -> dict[str, Any]:
    tensor_args_t = ctypes.c_void_p * 5
    scalar_args_t = ctypes.c_float * 4
    scalars = scalar_args_t(1.0, 0.0, 0.0, 0.0)

    submits = (
        CudaPtoTaskSubmit(
            "rmsnorm",
            CALLABLES[0][1],
            (
                CudaPtoTaskArg("hidden", ptr=ptrs["hidden"]),
                CudaPtoTaskArg("norm_weight", role="no_dep", ptr=ptrs["norm_weight"]),
                CudaPtoTaskArg("rmsnorm", role="output", ptr=ptrs["rmsnorm"]),
            ),
        ),
        CudaPtoTaskSubmit(
            "attention",
            CALLABLES[1][1],
            (
                CudaPtoTaskArg("rmsnorm", ptr=ptrs["rmsnorm"]),
                CudaPtoTaskArg("q_weight", role="no_dep", ptr=ptrs["q_weight"]),
                CudaPtoTaskArg("k_weight", role="no_dep", ptr=ptrs["k_weight"]),
                CudaPtoTaskArg("v_weight", role="no_dep", ptr=ptrs["v_weight"]),
                CudaPtoTaskArg("context", role="output", ptr=ptrs["context"]),
            ),
        ),
        CudaPtoTaskSubmit(
            "mlp",
            CALLABLES[2][1],
            (
                CudaPtoTaskArg("context", ptr=ptrs["context"]),
                CudaPtoTaskArg("gate_weight", role="no_dep", ptr=ptrs["gate_weight"]),
                CudaPtoTaskArg("up_weight", role="no_dep", ptr=ptrs["up_weight"]),
                CudaPtoTaskArg("mlp", role="output", ptr=ptrs["mlp"]),
            ),
        ),
        CudaPtoTaskSubmit(
            "final_norm",
            CALLABLES[3][1],
            (
                CudaPtoTaskArg("mlp", ptr=ptrs["mlp"]),
                CudaPtoTaskArg(
                    "final_norm_weight",
                    role="no_dep",
                    ptr=ptrs["final_norm_weight"],
                ),
                CudaPtoTaskArg("final_norm", role="output", ptr=ptrs["final_norm"]),
            ),
        ),
        CudaPtoTaskSubmit(
            "logits",
            CALLABLES[4][1],
            (
                CudaPtoTaskArg("final_norm", ptr=ptrs["final_norm"]),
                CudaPtoTaskArg("lm_head", role="no_dep", ptr=ptrs["lm_head"]),
                CudaPtoTaskArg("logits", role="output", ptr=ptrs["logits"]),
            ),
        ),
    )

    def make_task(node, dependent_begin, dependent_count, initial_fanin):
        assert node.attrs is not None
        submit = node.attrs["submit"]
        common = {
            "func_id": submit.callable_id,
            "n": 4,
            "dependent_begin": dependent_begin,
            "dependent_count": dependent_count,
            "initial_fanin": initial_fanin,
            "scalar_args": scalars,
            "scalar_arg_count": 1,
        }
        if node.key == "rmsnorm":
            return CudaPersistentDagTask(
                **common,
                a=ptrs["hidden"],
                out=ptrs["rmsnorm"],
                tensor_args=tensor_args_t(ptrs["norm_weight"], None, None, None, None),
                tensor_arg_count=1,
            )
        if node.key == "attention":
            return CudaPersistentDagTask(
                **common,
                a=ptrs["rmsnorm"],
                out=ptrs["context"],
                c=ptrs["key_cache"],
                d=ptrs["value_cache"],
                tensor_args=tensor_args_t(
                    ptrs["q_weight"],
                    ptrs["k_weight"],
                    ptrs["v_weight"],
                    None,
                ),
                tensor_arg_count=3,
            )
        if node.key == "mlp":
            return CudaPersistentDagTask(
                **common,
                a=ptrs["context"],
                out=ptrs["mlp"],
                tensor_args=tensor_args_t(
                    ptrs["gate_weight"], ptrs["up_weight"], None, None
                ),
                tensor_arg_count=2,
            )
        if node.key == "final_norm":
            return CudaPersistentDagTask(
                **common,
                a=ptrs["mlp"],
                out=ptrs["final_norm"],
                tensor_args=tensor_args_t(
                    ptrs["final_norm_weight"], None, None, None
                ),
                tensor_arg_count=1,
            )
        logits_common = {
            **common,
            "scalar_args": scalar_args_t(0.0, 4.0, 0.0, 0.0),
            "scalar_arg_count": 2,
        }
        return CudaPersistentDagTask(
            **logits_common,
            a=ptrs["final_norm"],
            out=ptrs["logits"],
            tensor_args=tensor_args_t(ptrs["lm_head"], None, None, None, None),
            tensor_arg_count=1,
        )

    lowered = lower_cuda_pto_task_graph(submits, make_task)
    return {
        "tasks": (CudaPersistentDagTask * len(lowered.tasks))(*lowered.tasks),
        "dependents": (ctypes.c_uint32 * len(lowered.dependents))(
            *lowered.dependents
        ),
        "fanin": (ctypes.c_uint32 * len(lowered.fanin))(*lowered.fanin),
    }


def make_tasks(ptrs: dict[str, int]) -> Any:
    return make_graph_arrays(ptrs)["tasks"]


def make_state(plan: dict[str, Any], ptrs: dict[str, int]) -> CudaPersistentDagState:
    word = ctypes.sizeof(ctypes.c_uint32)
    return CudaPersistentDagState(
        tasks=ptrs["tasks"],
        task_count=plan["dag"]["task_count"],
        dependents=ptrs["dependents"],
        dependent_count=plan["dag"]["dependent_count"],
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
