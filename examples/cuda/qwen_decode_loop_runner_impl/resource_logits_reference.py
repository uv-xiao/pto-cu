"""Logits summary and host-side diagnostic references for Qwen runs."""

from __future__ import annotations

import heapq
import math
import struct
from typing import Any


MAX_LOGITS_REFERENCE_WEIGHT_ELEMENTS = 1_000_000
MAX_LOGITS_REFERENCE_CHECKED_ELEMENTS = 65_536
PTO_CUDA_DTYPE_BFLOAT16 = 6


def logits_written_elements(workspace: dict[str, Any]) -> int:
    return int(workspace["logits_buffer"].get("element_count", 0))


def active_logits_written_elements(final_task: Any, workspace: dict[str, Any]) -> int:
    buffer_elements = logits_written_elements(workspace)
    cols = int(getattr(final_task, "cols", 0))
    task_elements = int(getattr(final_task, "n", 0))
    if buffer_elements <= 0 or cols <= 0 or task_elements <= 0:
        return buffer_elements
    requested_active_cols = int(float(getattr(final_task, "scalar1", 0.0)))
    active_cols = requested_active_cols if requested_active_cols > 0 else cols
    active_cols = min(active_cols, cols)
    rows = task_elements // cols
    return min(buffer_elements, rows * active_cols)


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


def diagnostic_logits_projection_values(
    *,
    hidden: list[float],
    lm_head: list[float],
    count: int | None = None,
    indices: list[int] | None = None,
    cols: int,
    hidden_width: int,
    hidden_stride: int,
    weight_stride: int,
) -> list[float]:
    values = []
    selected = indices if indices is not None else list(range(int(count or 0)))
    cols = max(1, int(cols))
    hidden_width = max(1, int(hidden_width))
    hidden_stride = max(hidden_width, int(hidden_stride))
    weight_stride = max(hidden_width, int(weight_stride))
    for index in selected:
        row = index // cols
        col = index % cols
        acc = 0.0
        for k in range(hidden_width):
            a_index = row * hidden_stride + k
            weight_index = col * weight_stride + k
            if a_index >= len(hidden) or weight_index >= len(lm_head):
                break
            acc += hidden[a_index] * lm_head[weight_index]
        values.append(acc)
    return values


def diagnostic_logits_fallback_values(
    *,
    hidden: list[float],
    lm_head: list[float],
    indices: list[int],
) -> list[float]:
    hidden_elements = max(1, len(hidden))
    weight_elements = max(1, len(lm_head))
    return [
        hidden[index % hidden_elements] * lm_head[(index & 3) % weight_elements]
        for index in indices
    ]


def diagnostic_logits_reference_indices(
    *,
    value_count: int,
    cols: int,
    hidden_width: int,
    weight_stride: int,
    max_weight_elements: int,
    max_checked_elements: int,
) -> list[int]:
    if value_count <= 0 or cols <= 0 or hidden_width <= 0 or weight_stride <= 0:
        return []
    max_weight_backed_cols = (
        max(0, int(max_weight_elements) - int(hidden_width))
        // int(weight_stride)
        + 1
    )
    checked = min(
        int(value_count),
        int(max_checked_elements),
        int(cols),
        max_weight_backed_cols,
    )
    return list(range(max(0, checked)))


def tensor_arg_values_to_f32(values: Any, *, dtype_code: int) -> list[float]:
    if dtype_code == PTO_CUDA_DTYPE_BFLOAT16:
        return [
            struct.unpack("f", struct.pack("I", int(value) << 16))[0]
            for value in values
        ]
    return [float(value) for value in values]


def compare_logits_reference(
    values: list[float],
    reference: list[float],
    *,
    checked_indices: list[int] | None = None,
    formula: str = (
        "out[row,col]=sum_k hidden[row*hidden_stride+k]"
        "*lm_head[col*weight_stride+k]"
    ),
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    max_abs_error = 0.0
    mismatch_count = 0
    selected_values = (
        [values[index] for index in checked_indices]
        if checked_indices is not None
        else values
    )
    for value, expected in zip(selected_values, reference, strict=True):
        error = abs(value - expected)
        max_abs_error = max(max_abs_error, error)
        if error > tolerance:
            mismatch_count += 1
    return {
        "status": "pass" if mismatch_count == 0 else "fail",
        "scope": "diagnostic_qwen_tiled_vocab_projection",
        "formula": formula,
        "checked_element_count": len(selected_values),
        "tolerance": tolerance,
        "max_abs_error": round(float(max_abs_error), 8),
        "mismatch_count": mismatch_count,
    }
