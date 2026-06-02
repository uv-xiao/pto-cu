"""Graph and oracle helpers for Qwen submission smoke execution."""

from __future__ import annotations

import ctypes
import math
from typing import Any

from simpler_setup.cuda_callable_compiler import (
    CudaPersistentDagState,
    CudaPersistentDagTask,
)

from qwen_decode_loop_runner_impl.submission import QWEN_TASK_FUNCTIONS


def smoke_inputs() -> dict[str, list[float] | list[int]]:
    return {
        "token_ids": [1, 2, 3, 0],
        "mask": [1.0, 1.0, 1.0, 1.0],
        "embedding": [0.1, 0.2, 0.3, 0.4],
        "input_norm": [1.0, 1.1, 1.2, 1.3],
        "q_proj": [0.5, 0.6, 0.7, 0.8],
        "k_proj": [0.9, 1.0, 1.1, 1.2],
        "v_proj": [1.3, 1.4, 1.5, 1.6],
        "q_norm": [0.7, 0.8, 0.9, 1.0],
        "k_norm": [1.1, 1.2, 1.3, 1.4],
        "o_proj": [0.2, 0.1, 0.0, -0.1],
        "post_norm": [1.0, 0.9, 0.8, 0.7],
        "gate_proj": [0.1, 0.2, 0.3, 0.4],
        "up_proj": [1.0, 1.1, 1.2, 1.3],
        "down_proj": [0.3, 0.2, 0.1, 0.0],
        "final_norm": [1.0, 1.0, 1.0, 1.0],
        "lm_head": [2.0, 2.1, 2.2, 2.3],
    }


def expected_outputs(
    inputs: dict[str, list[float] | list[int]],
) -> dict[str, list[float]]:
    token_ids = inputs["token_ids"]
    embedding = inputs["embedding"]
    x = [embedding[token & 3] for token in token_ids]
    scale = 1.0 / math.sqrt(sum(value * value for value in x) / len(x) + 0.000001)
    x = [value * scale * weight for value, weight in zip(x, inputs["input_norm"])]
    key_cache = [value * weight for value, weight in zip(x, inputs["k_proj"])]
    value_cache = [value * weight for value, weight in zip(x, inputs["v_proj"])]
    x = value_cache
    x = [
        value * 0.5 * (q_norm + k_norm)
        for value, q_norm, k_norm in zip(x, inputs["q_norm"], inputs["k_norm"])
    ]
    x = [value + weight for value, weight in zip(x, inputs["o_proj"])]
    x = [value * weight for value, weight in zip(x, inputs["post_norm"])]
    x = [
        gate / (1.0 + math.exp(-gate)) * up
        for gate, up in zip(inputs["gate_proj"], inputs["up_proj"])
    ]
    x = [value + weight for value, weight in zip(x, inputs["down_proj"])]
    x = [value * weight for value, weight in zip(x, inputs["final_norm"])]
    logits = [value * weight for value, weight in zip(x, inputs["lm_head"])]
    return {
        "logits": [round(value, 6) for value in logits],
        "key_cache": [round(value, 6) for value in key_cache],
        "value_cache": [round(value, 6) for value in value_cache],
    }


def make_tasks(ptrs: dict[str, int]) -> Any:
    tensor_args_t = ctypes.c_void_p * 5
    scalar_args_t = ctypes.c_float * 4
    task_t = CudaPersistentDagTask * 10
    scalars = scalar_args_t(1.0, 0.0, 0.0, 0.0)
    tensors = [
        tensor_args_t(ptrs["embedding"], None, None, None, None),
        tensor_args_t(ptrs["input_norm"], None, None, None, None),
        tensor_args_t(ptrs["q_proj"], ptrs["k_proj"], ptrs["v_proj"], None, None),
        tensor_args_t(ptrs["q_norm"], ptrs["k_norm"], None, None, None),
        tensor_args_t(ptrs["o_proj"], None, None, None, None),
        tensor_args_t(ptrs["post_norm"], None, None, None, None),
        tensor_args_t(ptrs["gate_proj"], ptrs["up_proj"], None, None, None),
        tensor_args_t(ptrs["down_proj"], None, None, None, None),
        tensor_args_t(ptrs["final_norm"], None, None, None, None),
        tensor_args_t(ptrs["lm_head"], None, None, None, None),
    ]
    outputs = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "logits"]
    inputs = ["token_ids", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
    counts = [1, 1, 3, 2, 1, 1, 2, 1, 1, 1]
    return task_t(
        *[
            CudaPersistentDagTask(
                func_id=func["func_id"],
                a=ptrs[inputs[index]],
                b=ptrs["mask"],
                out=ptrs[outputs[index]],
                n=4,
                dependent_begin=index,
                dependent_count=1 if index < 9 else 0,
                initial_fanin=0 if index == 0 else 1,
                c=ptrs["key_cache"],
                d=ptrs["value_cache"],
                tensor_args=tensors[index],
                scalar_args=scalars,
                tensor_arg_count=counts[index],
                scalar_arg_count=1,
            )
            for index, func in enumerate(QWEN_TASK_FUNCTIONS)
        ]
    )


def make_state(plan: dict[str, Any], ptrs: dict[str, int]) -> CudaPersistentDagState:
    word = ctypes.sizeof(ctypes.c_uint32)
    return CudaPersistentDagState(
        tasks=ptrs["tasks"],
        task_count=10,
        dependents=ptrs["dependents"],
        dependent_count=9,
        fanin=ptrs["fanin"],
        ready_queue=ptrs["ready_queue"],
        ready_flags=ptrs["ready_flags"],
        completion_queue=ptrs["completion_queue"],
        completion_flags=ptrs["completion_flags"],
        queue_capacity=plan["queue_capacity"],
        queue_head=ptrs["counters"],
        queue_tail=ptrs["counters"] + word,
        completion_head=ptrs["counters"] + 2 * word,
        completion_tail=ptrs["counters"] + 3 * word,
        completed_count=ptrs["counters"] + 4 * word,
        error_count=ptrs["counters"] + 5 * word,
        error_code=ptrs["counters"] + 6 * word,
        error_task_id=ptrs["counters"] + 7 * word,
        scheduler_blocks=plan["scheduler_blocks"],
        scheduler_init_count=ptrs["counters"] + 8 * word,
        scheduler_loop_count=ptrs["counters"] + 9 * word,
        scheduler_processed_count=ptrs["counters"] + 10 * word,
        scheduler_processed_by_block=ptrs["scheduler_processed"],
    )
