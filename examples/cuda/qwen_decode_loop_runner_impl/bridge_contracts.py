"""Qwen decode-loop diagnostic bridge contracts."""

from __future__ import annotations

from typing import Any


LIVE_MICRODECODE_ARTIFACT = (
    "tmp/cuda-backend/pto-serving-decode-loop-live-2026-06-01/"
    "qwen-microdecode-loop.json"
)
LIVE_UNIT_MATH_ARTIFACT = (
    "tmp/cuda-backend/qwen-final-norm-rmsnorm/"
    "qwen-unit-math-live.json"
)
LIVE_MICRODECODE_FIELDS = {
    "a": "hidden_state",
    "b": "attention_mask",
    "out": "logits_out",
    "c": "key_cache_mutable",
    "d": "value_cache_mutable",
    "tensor_args": "resident_weight_tensors",
}
LIVE_DECODE_LOOP_REUSE = {
    "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
    "reset_between_runs": [
        "fanin",
        "ready_flags",
        "completion_flags",
        "counters",
    ],
    "carried_between_runs": ["key_cache_mutable", "value_cache_mutable"],
}


def cuda_live_bridge_contract() -> dict[str, Any]:
    return {
        "status": "diagnostic_bridge_ready",
        "runtime": "cuda/persistent_device",
        "source_live_artifact": LIVE_MICRODECODE_ARTIFACT,
        "submission_to_live_fields": LIVE_MICRODECODE_FIELDS,
        "decode_loop_reuse": LIVE_DECODE_LOOP_REUSE,
        "serving_coverage": "diagnostic_microdecode",
        "remaining_gap": "full_qwen_decode_loop_execution",
    }


def unit_math_live_bridge_contract(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "status": "diagnostic_bridge_ready",
        "runtime": "cuda/persistent_device",
        "source_live_artifact": LIVE_UNIT_MATH_ARTIFACT,
        "submission_to_live_fields": {
            "a": "hidden_state",
            "out": "logits_out",
            "c": "key_cache_mutable",
            "d": "value_cache_mutable",
            "tensor_args": "unit_weight_tensors",
        },
        "decode_loop_reuse": {
            "prepared_callable_reuse": "single_prepare_multiple_run_prepared",
            "reset_between_runs": [
                "fanin",
                "ready_flags",
                "completion_flags",
                "counters",
                "unit_outputs",
            ],
            "carried_between_runs": [
                "hidden_state_from_previous_logits",
                "weight_buffers",
                "kv_cache_buffers",
            ],
        },
        "serving_coverage": "diagnostic_unit_math",
        "remaining_gap": "full_qwen_decode_loop_execution",
    }
    if payload is None:
        return contract
    summary = payload.get("decode_loop_summary", {})
    contract["status"] = "diagnostic_bridge_executed"
    contract["live_summary"] = {
        "status": payload.get("status", "unknown"),
        "repeat_runs": int(summary.get("repeat_runs", 1)),
        "total_completed_count": int(summary.get("total_completed_count", 0)),
        "total_error_count": int(summary.get("total_error_count", 0)),
        "max_abs_error": float(payload.get("max_abs_error", 0.0)),
    }
    return contract
