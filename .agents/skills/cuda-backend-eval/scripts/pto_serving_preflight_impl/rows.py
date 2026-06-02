"""Benchmark-viewer row gates for PTO Qwen full-serving evidence."""

from __future__ import annotations

from typing import Any

from pto_serving_preflight_impl.constants import (
    FULL_SERVING_METRIC_FIELDS,
    PAPER_WORKLOAD_IDS,
)


def pto_serving_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in results.get("result_records", []):
        if not isinstance(record, dict):
            continue
        if record.get("benchmark_id") != "llm_serving_decode":
            continue
        if record.get("method_id") != "pto_persistent_device":
            continue
        rows.append(record)
    return rows


def row_workload_id(row: dict[str, Any]) -> str:
    statistic = row.get("statistic", {})
    workload_id = statistic.get("workload_id")
    if isinstance(workload_id, str) and workload_id:
        return workload_id
    shape = str(row.get("inputs", {}).get("shape", ""))
    for candidate in PAPER_WORKLOAD_IDS:
        if candidate in shape:
            return candidate
    return ""


def full_serving_qwen_row_status(row: dict[str, Any]) -> dict[str, Any]:
    statistic = row.get("statistic", {})
    shape = str(row.get("inputs", {}).get("shape", ""))
    workload_id = row_workload_id(row)
    missing = []
    if row.get("benchmark_id") != "llm_serving_decode":
        missing.append("benchmark_id=llm_serving_decode")
    if row.get("method_id") != "pto_persistent_device":
        missing.append("method_id=pto_persistent_device")
    if "Qwen/Qwen3-8B" not in shape:
        missing.append("inputs.shape contains Qwen/Qwen3-8B")
    if statistic.get("serving_coverage") != "full_serving":
        missing.append("statistic.serving_coverage=full_serving")
    if row.get("correctness") != "pass":
        missing.append("correctness=pass")
    if workload_id not in PAPER_WORKLOAD_IDS:
        missing.append("workload_id is mpk_offline_decode or vdcores_offline_decode")
    for key in sorted(FULL_SERVING_METRIC_FIELDS):
        value = statistic.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            missing.append(f"statistic.{key}>0")
    if not row.get("raw_artifact"):
        missing.append("raw_artifact")
    return {
        "status": "pass" if not missing else "fail",
        "workload_id": workload_id,
        "shape": shape,
        "raw_artifact": row.get("raw_artifact", ""),
        "correctness": row.get("correctness", ""),
        "serving_coverage": statistic.get("serving_coverage", ""),
        "missing_requirements": missing,
    }


def full_serving_qwen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if full_serving_qwen_row_status(row)["status"] == "pass"
    ]


def serving_policy_summaries(serving_workloads: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for workload in serving_workloads.get("serving_workloads", []):
        if not isinstance(workload, dict):
            continue
        if workload.get("id") not in PAPER_WORKLOAD_IDS:
            continue
        model_policy = workload.get("model_policy", {})
        prompt_policy = workload.get("prompt_policy", {})
        decode_policy = workload.get("decode_policy", {})
        summaries.append(
            {
                "id": workload.get("id", ""),
                "primary_model": model_policy.get("primary_model", ""),
                "target_prompt_tokens": prompt_policy.get("target_prompt_tokens"),
                "decode_tokens": decode_policy.get("decode_tokens"),
                "batch_sizes": decode_policy.get("batch_sizes", []),
            }
        )
    return summaries
