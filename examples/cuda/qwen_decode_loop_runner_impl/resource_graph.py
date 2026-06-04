"""Device graph-state materialization for resource-backed Qwen runs."""

from __future__ import annotations

import ctypes
from typing import Any

from simpler_setup.cuda_callable_compiler import CudaPersistentDagState

from qwen_decode_loop_runner_impl.launch_preflight import next_power_of_two
from qwen_decode_loop_runner_impl.resource_logits_reference import (
    MAX_LOGITS_REFERENCE_CHECKED_ELEMENTS,
    MAX_LOGITS_REFERENCE_WEIGHT_ELEMENTS,
    PTO_CUDA_DTYPE_BFLOAT16,
    active_logits_cols,
    active_logits_sample_extent,
    active_logits_written_elements,
    compare_logits_reference,
    diagnostic_logits_fallback_values,
    diagnostic_logits_projection_values,
    diagnostic_logits_reference_row_count,
    diagnostic_logits_reference_indices,
    summarize_activation_row_values,
    summarize_logits_values,
    tensor_arg_values_to_f32,
)


def activation_buffer_index_for_packet_task(
    *,
    task_index: int,
    packet_index_offset: int = 0,
) -> int:
    return max(int(packet_index_offset), 0) + int(task_index)


class MaterializedGraph:
    def __init__(
        self,
        session: Any,
        packet: Any,
        *,
        scheduler_blocks: int = 1,
        block_dim: int = 64,
    ) -> None:
        self.session = session
        self.packet = packet
        self.task_count = len(packet)
        self.queue_capacity = next_power_of_two(max(self.task_count + 1, 16))
        self.scheduler_blocks = max(1, int(scheduler_blocks))
        self.block_dim = max(1, int(block_dim))
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
            "scheduler_processed": (u32 * self.scheduler_blocks)(
                *([0] * self.scheduler_blocks),
            ),
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
            scheduler_blocks=self.scheduler_blocks,
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
            "scheduler_processed_by_block": [
                int(self.hosts["scheduler_processed"][index])
                for index in range(self.scheduler_blocks)
            ],
        }

    def read_output_sample(self, workspace: dict[str, Any]) -> list[float]:
        host = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
        ptr = int(workspace["logits_buffer"]["device_ptr_hex"], 0)
        self.copy_from_device(host, ptr, "logits_sample")
        return [round(float(value), 6) for value in host]

    def read_logits_summary(self, workspace: dict[str, Any]) -> dict[str, Any]:
        final_task = self.packet[self.task_count - 1]
        written_elements = active_logits_written_elements(final_task, workspace)
        sampled_elements = active_logits_sample_extent(final_task, workspace)
        if sampled_elements <= 0:
            return {
                "status": "not_sampled",
                "reason": "no_written_logits",
                "logits_buffer_elements": int(
                    workspace["logits_buffer"].get("element_count", 0),
                ),
                "written_element_count": written_elements,
                "sampled_element_count": 0,
            }
        host = (ctypes.c_float * sampled_elements)(*([0.0] * sampled_elements))
        ptr = int(workspace["logits_buffer"]["device_ptr_hex"], 0)
        self.copy_from_device(host, ptr, "logits_written_buffer")
        values = [float(value) for value in host]
        return summarize_logits_values(
            values,
            logits_buffer_elements=int(workspace["logits_buffer"]["element_count"]),
            written_element_count=written_elements,
            vocab_cols=int(final_task.cols),
            diagnostic_reference=self.diagnostic_logits_reference(
                final_task=final_task,
                values=values,
            ),
        )

    def read_activation_finiteness_summary(
        self,
        workspace: dict[str, Any],
        descriptors: list[dict[str, Any]],
        *,
        row_index: int = 0,
        max_columns: int = 12288,
        packet_index_offset: int = 0,
        selected_columns: list[int] | None = None,
        row_dump_descriptor_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        summaries = []
        activation_buffers = workspace.get("activation_buffers", [])
        for task_index in range(self.task_count - 1):
            buffer_index = activation_buffer_index_for_packet_task(
                task_index=task_index,
                packet_index_offset=packet_index_offset,
            )
            if buffer_index >= len(activation_buffers):
                continue
            task = self.packet[task_index]
            cols = int(getattr(task, "cols", 0))
            n = int(getattr(task, "n", 0))
            if cols <= 0 or n <= 0:
                continue
            row_begin = max(0, int(row_index)) * cols
            if row_begin >= n:
                continue
            sample_count = min(cols, n - row_begin, max(1, int(max_columns)))
            host = (ctypes.c_float * sample_count)(*([0.0] * sample_count))
            ptr = int(activation_buffers[buffer_index]["device_ptr_hex"], 0)
            self.copy_from_device(
                host,
                ptr + row_begin * ctypes.sizeof(ctypes.c_float),
                f"activation_row_{task_index}",
            )
            descriptor = descriptors[task_index] if task_index < len(descriptors) else {}
            descriptor_id = descriptor.get("id")
            row_summary = summarize_activation_row_values(
                [float(value) for value in host],
                row_index=row_index,
                row_width=cols,
                selected_columns=selected_columns,
                include_values=(
                    row_dump_descriptor_ids is not None
                    and str(descriptor_id) in row_dump_descriptor_ids
                ),
            )
            row_summary.update(
                {
                    "task_index": task_index,
                    "activation_buffer_index": buffer_index,
                    "descriptor_id": descriptor_id,
                    "callable": descriptor.get("callable"),
                    "output_element_count": n,
                    "output_rows": n // cols,
                    "sampled_column_count": sample_count,
                    "column_sample_policy": (
                        "full_row" if sample_count == cols else "row_prefix"
                    ),
                },
            )
            summaries.append(row_summary)
        first_nonfinite = next(
            (item for item in summaries if item["nonfinite_count"] > 0),
            None,
        )
        return {
            "status": "sampled" if summaries else "not_sampled",
            "row_index": int(row_index),
            "sampled_task_count": len(summaries),
            "first_nonfinite_task": first_nonfinite,
            "tasks": summaries,
        }

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
        if int(final_task.cols) <= 0 or int(final_task.inner) <= 0:
            return self.diagnostic_logits_fallback_reference(
                final_task=final_task,
                values=values,
                hidden_elements=hidden_elements,
            )
        cols = int(final_task.cols)
        hidden_width = int(final_task.inner)
        hidden_stride = int(final_task.lda) if int(final_task.lda) > 0 else hidden_width
        weight_stride = int(final_task.ldb) if int(final_task.ldb) > 0 else hidden_width
        checked_indices = diagnostic_logits_reference_indices(
            value_count=len(values),
            cols=cols,
            active_cols=active_logits_cols(final_task),
            hidden_width=hidden_width,
            weight_stride=weight_stride,
            max_weight_elements=MAX_LOGITS_REFERENCE_WEIGHT_ELEMENTS,
            max_checked_elements=MAX_LOGITS_REFERENCE_CHECKED_ELEMENTS,
        )
        if not checked_indices:
            return {
                "status": "not_checked",
                "reason": "logits_projection_reference_too_large",
                "scope": "diagnostic_qwen_tiled_vocab_projection",
                "max_reference_weight_elements": (
                    MAX_LOGITS_REFERENCE_WEIGHT_ELEMENTS
                ),
            }
        max_col = max(index % cols for index in checked_indices)
        required_weight_elements = max_col * weight_stride + hidden_width
        hidden = (ctypes.c_float * hidden_elements)(*([0.0] * hidden_elements))
        self.copy_from_device(hidden, int(final_task.a), "diagnostic_logits_hidden")
        lm_head = self.copy_tensor_arg_to_f32(
            ptr=int(final_task.tensor_args[0]),
            dtype_code=int(final_task.tensor_arg_dtypes[0]),
            element_count=required_weight_elements,
            label="diagnostic_logits_lm_head",
        )
        reference = diagnostic_logits_projection_values(
            hidden=[float(value) for value in hidden],
            lm_head=lm_head,
            indices=checked_indices,
            cols=cols,
            hidden_width=hidden_width,
            hidden_stride=hidden_stride,
            weight_stride=weight_stride,
        )
        comparison = compare_logits_reference(
            values,
            reference,
            checked_indices=checked_indices,
        )
        comparison["checked_row_count"] = diagnostic_logits_reference_row_count(
            checked_indices=checked_indices,
            cols=cols,
        )
        return comparison

    def diagnostic_logits_fallback_reference(
        self,
        *,
        final_task: Any,
        values: list[float],
        hidden_elements: int,
    ) -> dict[str, Any]:
        checked_indices = list(
            range(min(len(values), MAX_LOGITS_REFERENCE_CHECKED_ELEMENTS)),
        )
        if not checked_indices:
            return {"status": "not_checked", "reason": "no_logits_to_check"}
        hidden = (ctypes.c_float * hidden_elements)(*([0.0] * hidden_elements))
        self.copy_from_device(hidden, int(final_task.a), "diagnostic_logits_hidden")
        lm_head = self.copy_tensor_arg_to_f32(
            ptr=int(final_task.tensor_args[0]),
            dtype_code=int(final_task.tensor_arg_dtypes[0]),
            element_count=4,
            label="diagnostic_logits_lm_head",
        )
        reference = diagnostic_logits_fallback_values(
            hidden=[float(value) for value in hidden],
            lm_head=lm_head,
            indices=checked_indices,
        )
        return compare_logits_reference(
            values,
            reference,
            checked_indices=checked_indices,
            formula=(
                "out[i]=hidden[i%hidden_elements]"
                "*lm_head[i&3]"
            ),
        )

    def copy_tensor_arg_to_f32(
        self,
        *,
        ptr: int,
        dtype_code: int,
        element_count: int,
        label: str,
    ) -> list[float]:
        if dtype_code == PTO_CUDA_DTYPE_BFLOAT16:
            host = (ctypes.c_uint16 * element_count)(*([0] * element_count))
            self.copy_from_device(host, ptr, label)
            return tensor_arg_values_to_f32(host, dtype_code=dtype_code)
        host = (ctypes.c_float * element_count)(*([0.0] * element_count))
        self.copy_from_device(host, ptr, label)
        return tensor_arg_values_to_f32(host, dtype_code=dtype_code)
