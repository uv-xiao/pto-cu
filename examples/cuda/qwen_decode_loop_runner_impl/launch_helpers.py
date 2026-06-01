"""Helper policies for Qwen resource-backed launch packets."""

from __future__ import annotations

from typing import Any


NUMERIC_TASK_MODES = ("diagnostic", "unit_math")
UNIT_NUMERIC_CALLABLES = {
    "qwen_attention_qkv",
    "qwen_mlp_gate_up",
}


def normalize_numeric_task_mode(mode: str) -> str:
    if mode not in NUMERIC_TASK_MODES:
        raise ValueError(f"unknown numeric task mode: {mode}")
    return mode


def attach_decode_feedback_tensors(
    *,
    index: int,
    task_count: int,
    descriptor: dict[str, Any],
    tensor_args: list[int],
    token_fields: dict[str, dict[str, Any]],
    workspace: dict[str, Any] | None,
) -> None:
    if (
        workspace is not None
        and is_logits_output_task(
            index=index,
            task_count=task_count,
            descriptor=descriptor,
        )
    ):
        tensor_args[2] = parse_ptr(token_fields["a"].get("device_ptr_hex"))
        tensor_args[3] = parse_ptr(token_fields["out"].get("device_ptr_hex"))


def tensor_arg_count(tensor_args: list[int]) -> int:
    for index in range(len(tensor_args) - 1, -1, -1):
        if tensor_args[index] != 0:
            return index + 1
    return 0


def set_decode_step_index(packet: Any, step_index: int | None) -> None:
    if step_index is None or not packet:
        return
    final_task = packet[len(packet) - 1]
    final_task.scalar_args[3] = float(step_index)
    final_task.scalar_arg_count = max(int(final_task.scalar_arg_count), 4)


def input_ptr_for_task(
    *,
    index: int,
    token_fields: dict[str, dict[str, Any]],
    workspace: dict[str, Any] | None,
) -> int:
    if workspace is not None and index > 0:
        return parse_ptr(workspace["activation_buffers"][index - 1]["device_ptr_hex"])
    return parse_ptr(token_fields["a"].get("device_ptr_hex"))


def output_ptr_for_task(
    *,
    index: int,
    task_count: int,
    token_fields: dict[str, dict[str, Any]],
    workspace: dict[str, Any] | None,
) -> int:
    if workspace is None:
        return parse_ptr(token_fields["out"].get("device_ptr_hex"))
    if index + 1 == task_count:
        return parse_ptr(workspace["logits_buffer"]["device_ptr_hex"])
    return parse_ptr(workspace["activation_buffers"][index]["device_ptr_hex"])


def task_n_for_record(
    *,
    index: int,
    task_count: int,
    descriptor: dict[str, Any],
    workspace: dict[str, Any] | None,
) -> int:
    if is_logits_output_task(index=index, task_count=task_count, descriptor=descriptor):
        return logits_element_count(workspace)
    return hidden_element_count(workspace)


def task_scalar_args(
    *,
    index: int,
    task_count: int,
    descriptor: dict[str, Any],
    workspace: dict[str, Any] | None,
    numeric_task_mode: str = "diagnostic",
) -> list[float]:
    if not is_logits_output_task(
        index=index,
        task_count=task_count,
        descriptor=descriptor,
    ):
        if (
            numeric_task_mode == "unit_math"
            and descriptor.get("callable") in UNIT_NUMERIC_CALLABLES
        ):
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0]
    return [
        0.0,
        float(hidden_element_count(workspace)),
        float(logits_element_count(workspace)),
        0.0,
    ]


def task_scalar_arg_count(scalar_args: list[float]) -> int:
    for index in range(len(scalar_args) - 1, -1, -1):
        if scalar_args[index] != 0.0:
            return index + 1
    return 0


def numeric_task_mode_summary(mode: str) -> dict[str, Any]:
    mode = normalize_numeric_task_mode(mode)
    return {
        "mode": mode,
        "numeric_ready_callables": sorted(UNIT_NUMERIC_CALLABLES)
        if mode == "unit_math"
        else [],
        "scope": (
            "resource_backed_unit_math_linear_branches"
            if mode == "unit_math"
            else "diagnostic_resource_backed_formulas"
        ),
    }


def is_logits_output_task(
    *,
    index: int,
    task_count: int,
    descriptor: dict[str, Any],
) -> bool:
    return index + 1 == task_count and descriptor.get("callable") == "qwen_logits"


def hidden_element_count(workspace: dict[str, Any] | None) -> int:
    if workspace is None:
        return 1
    buffers = workspace.get("activation_buffers", [])
    if buffers:
        return int(buffers[0].get("element_count", 1))
    return int(workspace["logits_buffer"].get("element_count", 1))


def logits_element_count(workspace: dict[str, Any] | None) -> int:
    if workspace is None:
        return 1
    return int(workspace["logits_buffer"].get("element_count", 1))


def tensor_arg_index(value: str) -> int:
    prefix = "tensor_args["
    if not value.startswith(prefix) or not value.endswith("]"):
        return 0
    parsed = int(value[len(prefix) : -1])
    return parsed if 0 <= parsed < 4 else 0


def parse_ptr(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        return int(value, 0)
    return 0


def missing_launch_buffers(
    *,
    descriptors: list[dict[str, Any]],
    workspace_ready: bool,
) -> list[dict[str, Any]]:
    if workspace_ready:
        return []
    return [
        {
            "buffer": "intermediate_activation_buffers",
            "required_count": max(len(descriptors) - 1, 0),
            "status": "not_allocated",
        },
        {
            "buffer": "float_logits_or_sampling_output",
            "required_count": 1,
            "status": "not_allocated",
        },
    ]


def launch_blockers(*, workspace_ready: bool) -> list[str]:
    if workspace_ready:
        return [
            "diagnostic_kernel_bodies_not_full_qwen_numeric",
            "run_prepared_execution_not_attempted",
        ]
    return [
        "intermediate_activation_buffers_not_allocated",
        "logits_output_dtype_mismatch_with_output_ids",
        "diagnostic_kernel_bodies_not_full_qwen_numeric",
    ]


def next_power_of_two(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power
