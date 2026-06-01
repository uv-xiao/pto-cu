"""Device graph-state materialization for resource-backed Qwen runs."""

from __future__ import annotations

import ctypes
import heapq
import math
from typing import Any

from simpler_setup.cuda_callable_compiler import CudaPersistentDagState

from qwen_decode_loop_runner_impl.launch_preflight import next_power_of_two


class MaterializedGraph:
    def __init__(self, session: Any, packet: Any) -> None:
        self.session = session
        self.packet = packet
        self.task_count = len(packet)
        self.queue_capacity = next_power_of_two(max(self.task_count + 1, 16))
        self.block_dim = 64
        self.ptrs: dict[str, int] = {}
        self.hosts = self.make_hosts()
        for name, host in self.hosts.items():
            self.ptrs[name] = self.alloc(ctypes.sizeof(host))
            self.copy_to_device(self.ptrs[name], host, name)
        self.ptrs["state"] = self.alloc(ctypes.sizeof(CudaPersistentDagState))
        self.copy_to_device(self.ptrs["state"], self.make_state(), "state")

    def make_hosts(self) -> dict[str, Any]:
        u32 = ctypes.c_uint32
        return {
            "tasks": self.packet,
            "dependents": (u32 * max(self.task_count - 1, 1))(
                *list(range(1, self.task_count)) or [0],
            ),
            "fanin": (u32 * self.task_count)(0, *([1] * (self.task_count - 1))),
            "ready_queue": (u32 * self.queue_capacity)(*([0] * self.queue_capacity)),
            "ready_flags": (u32 * self.queue_capacity)(*([0] * self.queue_capacity)),
            "completion_queue": (u32 * self.queue_capacity)(
                *([0] * self.queue_capacity),
            ),
            "completion_flags": (u32 * self.queue_capacity)(
                *([0] * self.queue_capacity),
            ),
            "counters": (u32 * 11)(*([0] * 11)),
            "scheduler_processed": (u32 * 1)(0),
        }

    def make_state(self) -> CudaPersistentDagState:
        word = ctypes.sizeof(ctypes.c_uint32)
        return CudaPersistentDagState(
            tasks=self.ptrs["tasks"],
            task_count=self.task_count,
            dependents=self.ptrs["dependents"],
            dependent_count=max(self.task_count - 1, 0),
            fanin=self.ptrs["fanin"],
            ready_queue=self.ptrs["ready_queue"],
            ready_flags=self.ptrs["ready_flags"],
            completion_queue=self.ptrs["completion_queue"],
            completion_flags=self.ptrs["completion_flags"],
            queue_capacity=self.queue_capacity,
            queue_head=self.ptrs["counters"],
            queue_tail=self.ptrs["counters"] + word,
            completion_head=self.ptrs["counters"] + 2 * word,
            completion_tail=self.ptrs["counters"] + 3 * word,
            completed_count=self.ptrs["counters"] + 4 * word,
            error_count=self.ptrs["counters"] + 5 * word,
            error_code=self.ptrs["counters"] + 6 * word,
            error_task_id=self.ptrs["counters"] + 7 * word,
            scheduler_blocks=1,
            scheduler_init_count=self.ptrs["counters"] + 8 * word,
            scheduler_loop_count=self.ptrs["counters"] + 9 * word,
            scheduler_processed_count=self.ptrs["counters"] + 10 * word,
            scheduler_processed_by_block=self.ptrs["scheduler_processed"],
        )

    def alloc(self, size: int) -> int:
        ptr = self.session.runtime.device_malloc_ctx(self.session.ctx, size)
        if not ptr:
            raise RuntimeError(f"device allocation failed for {size} bytes")
        value = int(ptr)
        self.session.allocations.append(("resource_backed_graph_runtime_state", value))
        return value

    def copy_to_device(self, ptr: int, host: Any, label: str) -> None:
        status = self.session.runtime.copy_to_device_ctx(
            self.session.ctx,
            ptr,
            ctypes.byref(host),
            ctypes.sizeof(host),
        )
        if status != 0:
            raise RuntimeError(f"copy_to_device {label} failed")

    def copy_from_device(self, host: Any, ptr: int, label: str) -> None:
        status = self.session.runtime.copy_from_device_ctx(
            self.session.ctx,
            ctypes.byref(host),
            ptr,
            ctypes.sizeof(host),
        )
        if status != 0:
            raise RuntimeError(f"copy_from_device {label} failed")

    def read_counters(self) -> dict[str, int | list[int]]:
        self.copy_from_device(self.hosts["counters"], self.ptrs["counters"], "counters")
        self.copy_from_device(
            self.hosts["scheduler_processed"],
            self.ptrs["scheduler_processed"],
            "scheduler_processed",
        )
        counters = self.hosts["counters"]
        return {
            "completed_count": int(counters[4]),
            "error_count": int(counters[5]),
            "error_code": int(counters[6]),
            "error_task_id": int(counters[7]),
            "scheduler_processed_count": int(counters[10]),
            "scheduler_processed_by_block": [int(self.hosts["scheduler_processed"][0])],
        }

    def read_output_sample(self, workspace: dict[str, Any]) -> list[float]:
        host = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
        ptr = int(workspace["logits_buffer"]["device_ptr_hex"], 0)
        self.copy_from_device(host, ptr, "logits_sample")
        return [round(float(value), 6) for value in host]

    def read_logits_summary(self, workspace: dict[str, Any]) -> dict[str, Any]:
        written_elements = logits_written_elements(workspace)
        checked_elements = written_elements
        if checked_elements <= 0:
            return {
                "status": "not_sampled",
                "reason": "no_written_logits",
                "logits_buffer_elements": int(
                    workspace["logits_buffer"].get("element_count", 0),
                ),
                "written_element_count": written_elements,
                "sampled_element_count": 0,
            }
        host = (ctypes.c_float * checked_elements)(*([0.0] * checked_elements))
        ptr = int(workspace["logits_buffer"]["device_ptr_hex"], 0)
        self.copy_from_device(host, ptr, "logits_written_buffer")
        values = [float(value) for value in host]
        final_task = self.packet[self.task_count - 1]
        return summarize_logits_values(
            values,
            logits_buffer_elements=int(workspace["logits_buffer"]["element_count"]),
            written_element_count=written_elements,
            diagnostic_reference=self.diagnostic_logits_reference(
                final_task=final_task,
                values=values,
            ),
        )

    def diagnostic_logits_reference(
        self,
        *,
        final_task: Any,
        values: list[float],
    ) -> dict[str, Any]:
        hidden_elements = int(final_task.scalar_args[1])
        if hidden_elements <= 0 or not final_task.a or not final_task.tensor_args[0]:
            return {
                "status": "not_checked",
                "reason": "missing_hidden_or_weight_pointer",
            }
        hidden = (ctypes.c_float * hidden_elements)(*([0.0] * hidden_elements))
        lm_head = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
        self.copy_from_device(hidden, int(final_task.a), "diagnostic_logits_hidden")
        self.copy_from_device(
            lm_head,
            int(final_task.tensor_args[0]),
            "diagnostic_logits_lm_head",
        )
        return compare_logits_formula(
            values,
            hidden=[float(value) for value in hidden],
            lm_head=[float(value) for value in lm_head],
        )


def logits_written_elements(workspace: dict[str, Any]) -> int:
    return int(workspace["logits_buffer"].get("element_count", 0))


def summarize_logits_values(
    values: list[float],
    *,
    logits_buffer_elements: int,
    written_element_count: int,
    diagnostic_reference: dict[str, Any] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    finite_values = [
        (index, value)
        for index, value in enumerate(values)
        if math.isfinite(value)
    ]
    ranked = heapq.nlargest(top_k, finite_values, key=lambda item: item[1])
    checksum = sum((index + 1) * value for index, value in enumerate(values))
    full_written = int(written_element_count) >= int(logits_buffer_elements)
    return {
        "status": "partial_logits_sampled",
        "coverage": (
            "full_logits_buffer_checked"
            if full_written and len(values) == int(logits_buffer_elements)
            else "full_logits_buffer_prefix_sampled"
            if full_written
            else "partial_logits_not_full_vocab"
        ),
        "logits_buffer_elements": int(logits_buffer_elements),
        "written_element_count": int(written_element_count),
        "sampled_element_count": len(values),
        "full_buffer_sampled": len(values) == int(logits_buffer_elements),
        "finite_count": len(finite_values),
        "nonzero_count": sum(1 for value in values if value != 0.0),
        "topk": [
            {
                "token_id": int(index),
                "logit": round(float(value), 6),
            }
            for index, value in ranked
        ],
        "sample_checksum": round(float(checksum), 6),
        "diagnostic_reference": diagnostic_reference
        or {"status": "not_checked", "reason": "not_requested"},
    }


def diagnostic_logits_reference_values(
    *,
    hidden: list[float],
    lm_head: list[float],
    count: int,
) -> list[float]:
    hidden_count = max(1, len(hidden))
    return [
        hidden[index % hidden_count] * lm_head[index & 3]
        for index in range(count)
    ]


def compare_logits_reference(
    values: list[float],
    reference: list[float],
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    max_abs_error = 0.0
    mismatch_count = 0
    for value, expected in zip(values, reference, strict=True):
        error = abs(value - expected)
        max_abs_error = max(max_abs_error, error)
        if error > tolerance:
            mismatch_count += 1
    return {
        "status": "pass" if mismatch_count == 0 else "fail",
        "scope": "diagnostic_qwen_logits_formula",
        "formula": "out[i]=hidden[i%hidden_elements]*lm_head[i&3]",
        "checked_element_count": len(values),
        "tolerance": tolerance,
        "max_abs_error": round(float(max_abs_error), 8),
        "mismatch_count": mismatch_count,
    }


def compare_logits_formula(
    values: list[float],
    *,
    hidden: list[float],
    lm_head: list[float],
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    hidden_count = max(1, len(hidden))
    max_abs_error = 0.0
    mismatch_count = 0
    for index, value in enumerate(values):
        expected = hidden[index % hidden_count] * lm_head[index & 3]
        error = abs(value - expected)
        max_abs_error = max(max_abs_error, error)
        if error > tolerance:
            mismatch_count += 1
    return {
        "status": "pass" if mismatch_count == 0 else "fail",
        "scope": "diagnostic_qwen_logits_formula",
        "formula": "out[i]=hidden[i%hidden_elements]*lm_head[i&3]",
        "checked_element_count": len(values),
        "tolerance": tolerance,
        "max_abs_error": round(float(max_abs_error), 8),
        "mismatch_count": mismatch_count,
    }
