"""Diagnostic sampled-token feedback for bounded Qwen decode steps."""

from __future__ import annotations

import ctypes
from typing import Any


I32_BYTES = ctypes.sizeof(ctypes.c_int32)


def apply_decode_feedback(
    *,
    session: Any,
    token_fields: dict[str, dict[str, Any]],
    decode_step_index: int | None,
    logits_summary: dict[str, Any],
) -> dict[str, Any]:
    token_id = sampled_token_id(logits_summary)
    if decode_step_index is None:
        return {"status": "not_requested"}
    if token_id is None:
        return {"status": "not_applied", "reason": "no_sampled_token"}
    if "out" not in token_fields or "a" not in token_fields:
        return {"status": "not_applied", "reason": "missing_token_fields"}

    output_ptr = parse_ptr(token_fields["out"].get("device_ptr_hex"))
    input_ptr = parse_ptr(token_fields["a"].get("device_ptr_hex"))
    output_index = int(decode_step_index)
    write_i32(session, output_ptr, output_index, token_id, "output_ids")
    write_i32(session, input_ptr, 0, token_id, "next_input_id")
    return {
        "status": "feedback_applied",
        "sampled_token_id": int(token_id),
        "output_ids_index": output_index,
        "output_ids_value": read_i32(session, output_ptr, output_index, "output_ids"),
        "next_input_index": 0,
        "next_input_value": read_i32(session, input_ptr, 0, "next_input_id"),
        "policy": "host_commits_diagnostic_sampled_token_for_next_step",
    }


def sampled_token_id(logits_summary: dict[str, Any]) -> int | None:
    topk = logits_summary.get("topk") or []
    if not topk:
        return None
    token_id = topk[0].get("token_id")
    return int(token_id) if token_id is not None else None


def write_i32(session: Any, base_ptr: int, index: int, value: int, label: str) -> None:
    host = ctypes.c_int32(int(value))
    status = session.runtime.copy_to_device_ctx(
        session.ctx,
        base_ptr + index * I32_BYTES,
        ctypes.byref(host),
        I32_BYTES,
    )
    if status != 0:
        raise RuntimeError(f"copy_to_device {label} feedback failed")


def read_i32(session: Any, base_ptr: int, index: int, label: str) -> int:
    host = ctypes.c_int32(0)
    status = session.runtime.copy_from_device_ctx(
        session.ctx,
        ctypes.byref(host),
        base_ptr + index * I32_BYTES,
        I32_BYTES,
    )
    if status != 0:
        raise RuntimeError(f"copy_from_device {label} feedback failed")
    return int(host.value)


def parse_ptr(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value:
        return int(value, 0)
    return 0


def feedback_summary(repeat_results: list[dict[str, Any]]) -> dict[str, Any]:
    feedback = [item.get("decode_feedback", {}) for item in repeat_results]
    applied = [item for item in feedback if item.get("status") == "feedback_applied"]
    if not feedback or all(item.get("status") == "not_requested" for item in feedback):
        return {"status": "not_requested"}
    return {
        "status": "diagnostic_token_feedback_applied"
        if len(applied) == len(feedback)
        else "partial_or_failed",
        "applied_step_count": len(applied),
        "step_count": len(feedback),
        "sampled_token_ids": [int(item["sampled_token_id"]) for item in applied],
        "policy": "host_commits_diagnostic_sampled_token_for_next_step",
    }
