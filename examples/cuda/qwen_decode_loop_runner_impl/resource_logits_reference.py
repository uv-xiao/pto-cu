"""Logits summary and host-side diagnostic references for Qwen runs."""

from __future__ import annotations

import heapq
import math
import struct
from typing import Any


MAX_LOGITS_REFERENCE_WEIGHT_ELEMENTS = 1_000_000
MAX_LOGITS_REFERENCE_CHECKED_ELEMENTS = 65_536
PTO_CUDA_DTYPE_BFLOAT16 = 6


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


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


def active_logits_sample_extent(final_task: Any, workspace: dict[str, Any]) -> int:
    buffer_elements = logits_written_elements(workspace)
    cols = int(getattr(final_task, "cols", 0))
    task_elements = int(getattr(final_task, "n", 0))
    if buffer_elements <= 0 or cols <= 0 or task_elements <= 0:
        return buffer_elements
    requested_active_cols = int(float(getattr(final_task, "scalar1", 0.0)))
    active_cols = requested_active_cols if requested_active_cols > 0 else cols
    active_cols = min(active_cols, cols)
    rows = task_elements // cols
    if rows <= 0 or active_cols <= 0:
        return 0
    extent = (rows - 1) * cols + active_cols
    return min(buffer_elements, extent)


def active_logits_cols(final_task: Any) -> int:
    cols = int(getattr(final_task, "cols", 0))
    if cols <= 0:
        return 0
    requested_active_cols = int(float(getattr(final_task, "scalar1", 0.0)))
    active_cols = requested_active_cols if requested_active_cols > 0 else cols
    return min(active_cols, cols)


def summarize_logits_values(
    values: list[float],
    *,
    logits_buffer_elements: int,
    written_element_count: int,
    diagnostic_reference: dict[str, Any] | None = None,
    vocab_cols: int | None = None,
    topk_row: int = 0,
    top_k: int = 5,
) -> dict[str, Any]:
    finite_values = finite_indexed_values(values)
    ranked = heapq.nlargest(
        top_k,
        topk_candidates(
            values,
            vocab_cols=vocab_cols,
            topk_row=topk_row,
        ),
        key=lambda item: item[1],
    )
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
                "token_id": int(token_id),
                "logit": round(float(value), 6),
            }
            for token_id, value in ranked
        ],
        "sample_checksum": round(float(checksum), 6),
        "diagnostic_reference": diagnostic_reference
        or {"status": "not_checked", "reason": "not_requested"},
    }


def finite_indexed_values(values: list[float]) -> list[tuple[int, float]]:
    return [
        (index, value)
        for index, value in enumerate(values)
        if math.isfinite(value)
    ]


def sample_activation_values(
    values: list[float],
    *,
    limit: int = 4,
) -> list[float | str]:
    sample: list[float | str] = []
    for value in values[: max(0, int(limit))]:
        if math.isnan(value):
            sample.append("nan")
        elif value == math.inf:
            sample.append("inf")
        elif value == -math.inf:
            sample.append("-inf")
        else:
            sample.append(round(float(value), 6))
    return sample


def summarize_activation_row_values(
    values: list[float],
    *,
    row_index: int,
    row_width: int,
) -> dict[str, Any]:
    first_nonfinite_column = None
    for column, value in enumerate(values):
        if not math.isfinite(value):
            first_nonfinite_column = column
            break
    finite_values = [value for value in values if math.isfinite(value)]
    finite_count = len(finite_values)
    nan_count = sum(1 for value in values if math.isnan(value))
    posinf_count = sum(1 for value in values if value == math.inf)
    neginf_count = sum(1 for value in values if value == -math.inf)
    first_nonfinite_index = (
        None
        if first_nonfinite_column is None
        else max(0, int(row_index)) * max(0, int(row_width)) + first_nonfinite_column
    )
    return {
        "row_index": int(row_index),
        "row_width": int(row_width),
        "sampled_element_count": len(values),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "nonfinite_count": len(values) - finite_count,
        "first_nonfinite_column": first_nonfinite_column,
        "first_nonfinite_index": first_nonfinite_index,
        "max_abs_finite": (
            round(float(max(abs(value) for value in finite_values)), 6)
            if finite_values
            else None
        ),
        "value_sample": sample_activation_values(values),
    }


def topk_candidates(
    values: list[float],
    *,
    vocab_cols: int | None,
    topk_row: int,
) -> list[tuple[int, float]]:
    if vocab_cols is None or int(vocab_cols) <= 0:
        return finite_indexed_values(values)
    cols = int(vocab_cols)
    row_begin = max(0, int(topk_row)) * cols
    row_end = min(row_begin + cols, len(values))
    return [
        (index - row_begin, value)
        for index, value in enumerate(values[row_begin:row_end], start=row_begin)
        if math.isfinite(value)
    ]


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
            acc = _float32(acc + hidden[a_index] * lm_head[weight_index])
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
    active_cols: int | None = None,
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
    row_count = max(1, math.ceil(int(value_count) / int(cols)))
    active_column_budget = (
        int(active_cols)
        if active_cols is not None and int(active_cols) > 0
        else int(cols)
    )
    column_budget = min(int(cols), active_column_budget, max_weight_backed_cols)
    checked = min(int(value_count), int(max_checked_elements))
    if checked <= 0 or column_budget <= 0:
        return []
    rows_to_check = min(row_count, checked)
    indices = []
    for col in range(column_budget):
        for row in range(rows_to_check):
            row_begin = row * int(cols)
            row_end = min(row_begin + int(cols), int(value_count))
            index = row_begin + col
            if index < row_end:
                indices.append(index)
            if len(indices) >= checked:
                return indices
    return indices


def diagnostic_logits_reference_row_count(
    *,
    checked_indices: list[int],
    cols: int,
) -> int:
    if cols <= 0:
        return 0
    return len({index // int(cols) for index in checked_indices})


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
    tolerance: float = 2e-5,
) -> dict[str, Any]:
    max_abs_error = 0.0
    max_error_index: int | None = None
    max_error_value = 0.0
    max_error_expected = 0.0
    max_error_allowed_error = 0.0
    mismatch_count = 0
    selected_values = (
        [values[index] for index in checked_indices]
        if checked_indices is not None
        else values
    )
    selected_indices = (
        checked_indices
        if checked_indices is not None
        else list(range(len(selected_values)))
    )
    first_mismatch: dict[str, Any] = {}
    for original_index, value, expected in zip(
        selected_indices,
        selected_values,
        reference,
        strict=True,
    ):
        error = abs(value - expected)
        allowed_error = float(tolerance)
        if error > max_abs_error:
            max_abs_error = error
            max_error_index = int(original_index)
            max_error_value = value
            max_error_expected = expected
            max_error_allowed_error = allowed_error
        if error > allowed_error:
            mismatch_count += 1
            if not first_mismatch:
                first_mismatch = {
                    "first_mismatch_index": int(original_index),
                    "first_mismatch_error": round(float(error), 8),
                    "first_mismatch_value": round(float(value), 8),
                    "first_mismatch_expected": round(float(expected), 8),
                    "first_mismatch_allowed_error": round(
                        float(allowed_error),
                        8,
                    ),
                }
    result = {
        "status": "pass" if mismatch_count == 0 else "fail",
        "scope": "diagnostic_qwen_tiled_vocab_projection",
        "formula": formula,
        "checked_element_count": len(selected_values),
        "tolerance": tolerance,
        "max_abs_error": round(float(max_abs_error), 8),
        "max_error_index": max_error_index,
        "max_error_value": round(float(max_error_value), 8),
        "max_error_expected": round(float(max_error_expected), 8),
        "max_error_allowed_error": round(float(max_error_allowed_error), 8),
        "mismatch_count": mismatch_count,
    }
    result.update(first_mismatch)
    return result
