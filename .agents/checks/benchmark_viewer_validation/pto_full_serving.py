from __future__ import annotations

import math
from typing import Any

from .common import fail


PTO_FULL_SERVING_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
PTO_FULL_SERVING_CORRECTNESS_SCOPE = "full_qwen_numerical_correctness"
PTO_FULL_SERVING_MODEL_ID = "Qwen/Qwen3-8B"
PTO_FULL_SERVING_MIN_SAMPLE_COUNT = 3


def validate_pto_full_serving_result(
    record: dict[str, Any],
    statistic: dict[str, Any],
    owner: str,
) -> None:
    if record.get("correctness") != "pass":
        fail(f"{owner} PTO full-serving row must pass correctness")
    shape = str(record.get("inputs", {}).get("shape", ""))
    workload_id = statistic.get("workload_id")
    if not isinstance(workload_id, str) or not workload_id:
        workload_id = next(
            (
                candidate
                for candidate in PTO_FULL_SERVING_WORKLOAD_IDS
                if candidate in shape
            ),
            "",
        )
    if workload_id not in PTO_FULL_SERVING_WORKLOAD_IDS:
        fail(f"{owner} PTO full-serving row has invalid workload_id")

    for key in (
        "host_wall_ns",
        "device_wall_ns",
        "end_to_end_latency_ns",
        "time_to_first_token_ns",
        "inter_token_latency_ns",
        "throughput_tokens_per_s",
    ):
        if not positive_number(statistic.get(key)):
            fail(f"{owner} has invalid statistic.{key}")

    batch_size = require_positive_int_stat(statistic, "batch_size", owner)
    prompt_tokens = require_positive_int_stat(statistic, "prompt_tokens", owner)
    decode_tokens = require_positive_int_stat(statistic, "decode_tokens", owner)
    sample_count = require_positive_int_stat(statistic, "sample_count", owner)
    if sample_count < PTO_FULL_SERVING_MIN_SAMPLE_COUNT:
        fail(
            f"{owner} sample_count must be at least "
            f"{PTO_FULL_SERVING_MIN_SAMPLE_COUNT}"
        )

    expected_input_tokens = batch_size * prompt_tokens
    expected_output_tokens = batch_size * decode_tokens
    if statistic.get("failed_requests") != 0:
        fail(f"{owner} failed_requests must be zero")
    exact_stat(statistic, "completed_requests", batch_size, owner)
    exact_stat(statistic, "total_input_tokens", expected_input_tokens, owner)
    exact_stat(statistic, "total_output_tokens", expected_output_tokens, owner)
    validate_pto_full_serving_correctness(
        record,
        statistic,
        expected_output_tokens,
        owner,
    )


def validate_pto_full_serving_correctness(
    record: dict[str, Any],
    statistic: dict[str, Any],
    expected_output_tokens: int,
    owner: str,
) -> None:
    details = record.get("correctness_details")
    if not isinstance(details, dict):
        fail(f"{owner} missing correctness_details")
    if details.get("scope") != PTO_FULL_SERVING_CORRECTNESS_SCOPE:
        fail(f"{owner} has invalid correctness_details.scope")
    if details.get("model_id") != PTO_FULL_SERVING_MODEL_ID:
        fail(f"{owner} has invalid correctness_details.model_id")
    if details.get("status") != "pass":
        fail(f"{owner} has invalid correctness_details.status")
    if details.get("token_match") is not True:
        fail(f"{owner} has invalid correctness_details.token_match")

    checked_token_count = require_positive_int_stat(
        details,
        "checked_token_count",
        owner,
    )
    if checked_token_count < expected_output_tokens:
        fail(
            f"{owner} checked_token_count must cover generated tokens: "
            f"expected at least {expected_output_tokens}, got {checked_token_count}"
        )

    max_abs_error = details.get("max_abs_error")
    tolerance = details.get("tolerance")
    if not nonnegative_number(max_abs_error):
        fail(f"{owner} has invalid correctness_details.max_abs_error")
    if not positive_number(tolerance):
        fail(f"{owner} has invalid correctness_details.tolerance")
    if max_abs_error > tolerance:
        fail(f"{owner} exceeds correctness_details.tolerance")

    if statistic.get("correctness_scope") != PTO_FULL_SERVING_CORRECTNESS_SCOPE:
        fail(f"{owner} has invalid statistic.correctness_scope")
    exact_stat(statistic, "checked_token_count", checked_token_count, owner)
    if statistic.get("max_abs_error") != max_abs_error:
        fail(f"{owner} has invalid statistic.max_abs_error")
    if statistic.get("correctness_tolerance") != tolerance:
        fail(f"{owner} has invalid statistic.correctness_tolerance")


def require_positive_int_stat(
    record: dict[str, Any],
    key: str,
    owner: str,
) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{owner} has invalid statistic.{key}")
    return value


def exact_stat(
    record: dict[str, Any],
    key: str,
    expected: int,
    owner: str,
) -> None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        fail(
            f"{owner} has invalid statistic.{key}: "
            f"expected {expected}, got {value}"
        )


def positive_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def nonnegative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )
