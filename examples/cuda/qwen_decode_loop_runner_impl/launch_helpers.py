"""Helper policies for Qwen resource-backed launch packets."""

from __future__ import annotations

from typing import Any


NUMERIC_TASK_MODES = (
    "diagnostic",
    "unit_math",
    "unit_math_full_rmsnorm",
)
UNIT_NUMERIC_CALLABLES = {
    "qwen_rmsnorm_input",
    "qwen_attention_qkv",
    "qwen_attention_qk_norm",
    "qwen_attention_o",
    "qwen_rmsnorm_post_attention",
    "qwen_mlp_gate_up",
    "qwen_mlp_down",
    "qwen_final_norm",
}
UNIT_NUMERIC_RMSNORM_SCALE = 1.0
UNIT_NUMERIC_ATTENTION_O_PROJECTION_INPUTS = 64.0
UNIT_NUMERIC_WEIGHTED_ELEMENTWISE_CALLABLES = {
    "qwen_attention_qk_norm",
    "qwen_attention_o",
    "qwen_rmsnorm_post_attention",
    "qwen_mlp_down",
    "qwen_final_norm",
}
FULL_RMSNORM_CALLABLES = {
    "qwen_rmsnorm_input",
    "qwen_rmsnorm_post_attention",
    "qwen_final_norm",
}
TASK_SHAPE_FIELDS = (
    "scalar0",
    "scalar1",
    "rows",
    "cols",
    "inner",
    "lda",
    "ldb",
    "ldc",
    "a_batch_stride",
    "b_batch_stride",
    "out_batch_stride",
)
FLOAT_TASK_SHAPE_FIELDS = {"scalar0", "scalar1"}
TENSOR_DTYPE_CODES = {
    "float32": 0,
    "bfloat16": 6,
}
PERSISTENT_TENSOR_ARG_CAPACITY = 5
QWEN_ATTENTION_O_FUNC_ID = 7104
LAYER_INDEX_SCALAR_ARG = 3
TERMINAL_LOGITS_BUFFER_CALLABLES = {"qwen_final_norm", "qwen_logits"}


def normalize_numeric_task_mode(mode: str) -> str:
    if mode not in NUMERIC_TASK_MODES:
        raise ValueError(f"unknown numeric task mode: {mode}")
    return mode


def is_unit_numeric_mode(mode: str) -> bool:
    return normalize_numeric_task_mode(mode) != "diagnostic"


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


def attach_mlp_down_residual_tensor(
    *,
    index: int,
    descriptor: dict[str, Any],
    descriptors: list[dict[str, Any]],
    tensor_args: list[int],
    token_fields: dict[str, dict[str, Any]],
    workspace: dict[str, Any] | None,
) -> None:
    if descriptor.get("callable") != "qwen_mlp_down" or len(tensor_args) < 2:
        return
    if tensor_args[1]:
        return
    input_norm_index = layer_input_norm_index(descriptors=descriptors, index=index)
    if input_norm_index is None:
        tensor_args[1] = parse_ptr(token_fields["a"].get("device_ptr_hex"))
        return
    tensor_args[1] = input_ptr_for_task(
        index=input_norm_index,
        token_fields=token_fields,
        workspace=workspace,
    )


def tensor_arg_count(tensor_args: list[int]) -> int:
    for index in range(len(tensor_args) - 1, -1, -1):
        if tensor_args[index] != 0:
            return index + 1
    return 0


def set_decode_step_index(packet: Any, step_index: int | None) -> None:
    if step_index is None or not packet:
        return
    final_task = packet[len(packet) - 1]
    final_task.scalar_args[LAYER_INDEX_SCALAR_ARG] = float(step_index)
    final_task.scalar_arg_count = max(
        int(final_task.scalar_arg_count),
        LAYER_INDEX_SCALAR_ARG + 1,
    )


def set_decode_step_state(
    packet: Any,
    *,
    step_index: int | None,
    decode_position: int | None,
) -> None:
    if not packet:
        return
    if decode_position is not None:
        kv_window = max(int(decode_position) + 1, 1)
        for task in packet:
            task.scalar_args[2] = float(decode_position)
            task.scalar_arg_count = max(int(task.scalar_arg_count), 3)
            if int(task.func_id) == QWEN_ATTENTION_O_FUNC_ID:
                task.inner = kv_window
    set_decode_step_index(packet, step_index)


def input_ptr_for_task(
    *,
    index: int,
    token_fields: dict[str, dict[str, Any]],
    workspace: dict[str, Any] | None,
) -> int:
    if workspace is not None and index > 0:
        return parse_ptr(workspace["activation_buffers"][index - 1]["device_ptr_hex"])
    return parse_ptr(token_fields["a"].get("device_ptr_hex"))


def residual_ptr_for_task(
    *,
    index: int,
    descriptors: list[dict[str, Any]],
    token_fields: dict[str, dict[str, Any]],
    workspace: dict[str, Any] | None,
) -> int:
    descriptor = descriptors[index]
    callable_name = descriptor.get("callable")
    if callable_name == "qwen_mlp_down":
        residual_index = layer_post_attention_norm_index(
            descriptors=descriptors,
            index=index,
        )
        if residual_index is not None:
            return input_ptr_for_task(
                index=residual_index,
                token_fields=token_fields,
                workspace=workspace,
            )
    if callable_name != "qwen_rmsnorm_post_attention":
        return parse_ptr(token_fields["b"].get("device_ptr_hex"))
    residual_index = layer_input_norm_index(
        descriptors=descriptors,
        index=index,
    )
    if residual_index is None:
        return parse_ptr(token_fields["a"].get("device_ptr_hex"))
    return input_ptr_for_task(
        index=residual_index,
        token_fields=token_fields,
        workspace=workspace,
    )


def layer_input_norm_index(
    *,
    descriptors: list[dict[str, Any]],
    index: int,
) -> int | None:
    descriptor_id = str(descriptors[index].get("id", ""))
    layer_prefix = descriptor_id
    for suffix in ("_post_attention_norm", "_mlp_down"):
        if descriptor_id.endswith(suffix):
            layer_prefix = descriptor_id.rsplit(suffix, 1)[0]
            break
    for candidate_index in range(index - 1, -1, -1):
        candidate = descriptors[candidate_index]
        if candidate.get("callable") != "qwen_rmsnorm_input":
            continue
        candidate_id = str(candidate.get("id", ""))
        if not layer_prefix or candidate_id.startswith(layer_prefix):
            return candidate_index
    return None


def layer_post_attention_norm_index(
    *,
    descriptors: list[dict[str, Any]],
    index: int,
) -> int | None:
    descriptor_id = str(descriptors[index].get("id", ""))
    layer_prefix = descriptor_id.rsplit("_mlp_down", 1)[0]
    for candidate_index in range(index - 1, -1, -1):
        candidate = descriptors[candidate_index]
        if candidate.get("callable") != "qwen_rmsnorm_post_attention":
            continue
        candidate_id = str(candidate.get("id", ""))
        if not layer_prefix or candidate_id.startswith(layer_prefix):
            return candidate_index
    return None


def output_ptr_for_task(
    *,
    index: int,
    absolute_index: int | None = None,
    task_count: int,
    descriptor: dict[str, Any] | None = None,
    token_fields: dict[str, dict[str, Any]],
    workspace: dict[str, Any] | None,
) -> int:
    if workspace is None:
        return parse_ptr(token_fields["out"].get("device_ptr_hex"))
    if index + 1 == task_count and terminal_task_writes_logits(descriptor):
        return parse_ptr(workspace["logits_buffer"]["device_ptr_hex"])
    target_index = index if absolute_index is None else absolute_index
    return parse_ptr(workspace["activation_buffers"][target_index]["device_ptr_hex"])


def task_n_for_record(
    *,
    index: int,
    absolute_index: int | None = None,
    task_count: int,
    descriptor: dict[str, Any],
    workspace: dict[str, Any] | None,
    numeric_task_mode: str = "diagnostic",
) -> int:
    if is_logits_output_task(index=index, task_count=task_count, descriptor=descriptor):
        return logits_element_count(workspace)
    target_index = index if absolute_index is None else absolute_index
    return task_output_element_count(index=target_index, workspace=workspace)


def task_scalar_args(
    *,
    index: int,
    absolute_index: int | None = None,
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
        layer_index = float(descriptor.get("layer_index", 0))
        if (
            is_unit_numeric_mode(numeric_task_mode)
            and descriptor.get("callable") in UNIT_NUMERIC_CALLABLES
        ):
            if descriptor.get("callable") == "qwen_rmsnorm_input":
                if numeric_task_mode == "unit_math_full_rmsnorm":
                    return [1.0, 0.0, 0.0, layer_index]
                return [1.0, UNIT_NUMERIC_RMSNORM_SCALE, 0.0, layer_index]
            if descriptor.get("callable") == "qwen_rmsnorm_post_attention":
                if numeric_task_mode == "unit_math_full_rmsnorm":
                    return [1.0, 0.0, 0.0, layer_index]
                return [1.0, UNIT_NUMERIC_RMSNORM_SCALE, 0.0, layer_index]
            if descriptor.get("callable") == "qwen_attention_o":
                projection_input_count = float(
                    descriptor.get(
                        "attention_o_projection_input_count",
                        UNIT_NUMERIC_ATTENTION_O_PROJECTION_INPUTS,
                    )
                )
                return [
                    1.0,
                    projection_input_count,
                    0.0,
                    layer_index,
                ]
            if descriptor.get("callable") == "qwen_final_norm":
                if numeric_task_mode == "unit_math_full_rmsnorm":
                    return [1.0, 0.0, 0.0, 0.0]
                return [1.0, UNIT_NUMERIC_RMSNORM_SCALE, 0.0, 0.0]
            return [1.0, 0.0, 0.0, layer_index]
        return [0.0, 0.0, 0.0, layer_index]
    return [
        0.0,
        float(
            task_input_element_count(
                index=index if absolute_index is None else absolute_index,
                workspace=workspace,
            )
        ),
        float(logits_element_count(workspace)),
        0.0,
    ]


def task_scalar_arg_count(scalar_args: list[float]) -> int:
    for index in range(len(scalar_args) - 1, -1, -1):
        if scalar_args[index] != 0.0:
            return index + 1
    return 0


def minimum_scalar_arg_count(
    *,
    descriptor: dict[str, Any],
    numeric_task_mode: str,
) -> int:
    if (
        numeric_task_mode == "unit_math_full_rmsnorm"
        and descriptor.get("callable") in FULL_RMSNORM_CALLABLES
    ):
        return 2
    return 0


def task_shape_fields(
    *,
    descriptor: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    fields: dict[str, int | float] = {}
    for source in (
        defaults or {},
        descriptor.get("task_shape_fields", {}),
        descriptor.get("scalar_fields", {}),
    ):
        for name, value in source.items():
            if name not in TASK_SHAPE_FIELDS or value is None:
                continue
            if name in FLOAT_TASK_SHAPE_FIELDS:
                fields[name] = float(value)
            else:
                fields[name] = int(value)
    return fields


def numeric_task_mode_summary(mode: str) -> dict[str, Any]:
    mode = normalize_numeric_task_mode(mode)
    if mode == "unit_math_full_rmsnorm":
        scope = "resource_backed_unit_math_full_rmsnorm_reduction"
    elif mode == "unit_math":
        scope = "resource_backed_unit_math_weighted_elementwise_branches"
    else:
        scope = "diagnostic_resource_backed_formulas"
    return {
        "mode": mode,
        "numeric_ready_callables": sorted(UNIT_NUMERIC_CALLABLES)
        if is_unit_numeric_mode(mode)
        else [],
        "external_scale_contracts": [
            {
                "callable": "qwen_rmsnorm_input",
                "scale_arg": "scalar_args[1]",
                "scale": UNIT_NUMERIC_RMSNORM_SCALE,
                "scope": "resource_backed_external_rmsnorm_scale",
            },
            {
                "callable": "qwen_rmsnorm_post_attention",
                "scale_arg": "scalar_args[1]",
                "scale": UNIT_NUMERIC_RMSNORM_SCALE,
                "scope": "resource_backed_external_rmsnorm_scale",
            },
        ]
        if mode == "unit_math"
        else [],
        "full_reduction_contracts": [
            {
                "callable": "qwen_rmsnorm_input",
                "scalar_arg_count": 2,
                "scope": "resource_backed_full_rmsnorm_reduction",
                "threading": "block",
            },
            {
                "callable": "qwen_rmsnorm_post_attention",
                "scalar_arg_count": 2,
                "scope": "resource_backed_full_rmsnorm_reduction",
                "threading": "block",
            },
            {
                "callable": "qwen_final_norm",
                "scalar_arg_count": 2,
                "scope": "resource_backed_full_rmsnorm_reduction",
                "threading": "block",
            },
        ]
        if mode == "unit_math_full_rmsnorm"
        else [],
        "weighted_elementwise_callables": sorted(
            UNIT_NUMERIC_WEIGHTED_ELEMENTWISE_CALLABLES,
        )
        if is_unit_numeric_mode(mode)
        else [],
        "scope": scope,
    }


def is_logits_output_task(
    *,
    index: int,
    task_count: int,
    descriptor: dict[str, Any],
) -> bool:
    return index + 1 == task_count and descriptor.get("callable") == "qwen_logits"


def hidden_element_count(
    workspace: dict[str, Any] | None,
    *,
    numeric_task_mode: str = "diagnostic",
) -> int:
    if workspace is None:
        return 1
    buffers = workspace.get("activation_buffers", [])
    if buffers:
        elements = int(buffers[0].get("element_count", 1))
    else:
        elements = int(workspace["logits_buffer"].get("element_count", 1))
    return elements


def task_input_element_count(*, index: int, workspace: dict[str, Any] | None) -> int:
    if workspace is None:
        return 1
    if index > 0:
        return activation_buffer_element_count(workspace=workspace, index=index - 1)
    return hidden_element_count(workspace)


def task_output_element_count(*, index: int, workspace: dict[str, Any] | None) -> int:
    if workspace is None:
        return 1
    return activation_buffer_element_count(workspace=workspace, index=index)


def activation_buffer_element_count(
    *,
    workspace: dict[str, Any],
    index: int,
) -> int:
    buffers = workspace.get("activation_buffers", [])
    if index < len(buffers):
        return int(buffers[index].get("element_count", 1))
    return hidden_element_count(workspace)


def logits_element_count(workspace: dict[str, Any] | None) -> int:
    if workspace is None:
        return 1
    return int(workspace["logits_buffer"].get("element_count", 1))


def tensor_arg_index(value: str) -> int:
    prefix = "tensor_args["
    if not value.startswith(prefix) or not value.endswith("]"):
        return 0
    parsed = int(value[len(prefix) : -1])
    return parsed if 0 <= parsed < PERSISTENT_TENSOR_ARG_CAPACITY else 0


def tensor_arg_dtype_codes(descriptor: dict[str, Any]) -> list[int]:
    codes = [0] * PERSISTENT_TENSOR_ARG_CAPACITY
    for item in descriptor.get("tensor_arg_metadata", []):
        if not isinstance(item, dict):
            continue
        index = tensor_arg_index(str(item.get("arg", "")))
        codes[index] = TENSOR_DTYPE_CODES.get(str(item.get("dtype", "float32")), 0)
    return codes


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
            "required_count": required_activation_buffer_count(descriptors),
            "status": "not_allocated",
        },
        {
            "buffer": "float_logits_or_sampling_output",
            "required_count": 1,
            "status": "not_allocated",
        },
    ]


def terminal_task_writes_logits(descriptor: dict[str, Any] | None) -> bool:
    return (
        descriptor is not None
        and descriptor.get("callable") in TERMINAL_LOGITS_BUFFER_CALLABLES
    )


def required_activation_buffer_count(descriptors: list[dict[str, Any]]) -> int:
    if not descriptors:
        return 0
    if terminal_task_writes_logits(descriptors[-1]):
        return max(len(descriptors) - 1, 0)
    return len(descriptors)


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
