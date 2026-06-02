"""Build the PTO Qwen full-serving preflight payload."""

from __future__ import annotations

from typing import Any

from viewer_data_io import load_json as load_viewer_json

from pto_serving_preflight_impl.checks import build_preflight_checks
from pto_serving_preflight_impl.constants import PAPER_WORKLOAD_IDS, VIEWER_DATA
from pto_serving_preflight_impl.io_helpers import (
    git_commit,
    load_json,
    load_serving_scaffold,
)
from pto_serving_preflight_impl.rows import (
    full_serving_qwen_row_status,
    full_serving_qwen_rows,
    pto_serving_rows,
    row_workload_id,
    serving_policy_summaries,
)


def build_preflight() -> dict[str, Any]:
    serving_workloads = load_json(VIEWER_DATA / "serving_workloads.json")
    results = load_viewer_json(VIEWER_DATA / "results.json")
    serving_scaffold = load_serving_scaffold()
    pto_rows = pto_serving_rows(results)
    qwen8b_pto_rows = full_serving_qwen_rows(pto_rows)
    qwen8b_row_statuses = [
        full_serving_qwen_row_status(row)
        for row in pto_rows
        if "Qwen/Qwen3-8B" in str(row.get("inputs", {}).get("shape", ""))
    ]
    qwen8b_present_workloads = sorted(
        {row_workload_id(row) for row in qwen8b_pto_rows}
    )
    qwen8b_missing_workloads = sorted(
        PAPER_WORKLOAD_IDS - set(qwen8b_present_workloads)
    )
    proxy_rows = [
        row
        for row in pto_rows
        if "attention tile proxy" in str(row.get("inputs", {}).get("shape", ""))
    ]
    checks = build_preflight_checks(
        serving_scaffold=serving_scaffold,
        proxy_rows=proxy_rows,
        qwen8b_missing_workloads=qwen8b_missing_workloads,
        qwen8b_present_workloads=qwen8b_present_workloads,
        qwen8b_row_statuses=qwen8b_row_statuses,
    )
    return {
        "schema_version": 1,
        "kind": "pto_persistent_device_full_serving_preflight",
        "status": "partial" if blocking_gaps(checks) else "pass",
        "commit": git_commit(),
        "serving_workloads": serving_policy_summaries(serving_workloads),
        "serving_lifecycle": serving_scaffold,
        "pto_serving_rows": row_summaries(pto_rows),
        "checks": checks,
        "blocking_gaps": blocking_gaps(checks),
        "next_action": (
            "Implement and import PTO persistent-device Qwen/Qwen3-8B "
            "full-serving rows for mpk_offline_decode and vdcores_offline_decode."
        ),
    }


def row_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "shape": row.get("inputs", {}).get("shape", ""),
            "raw_artifact": row.get("raw_artifact", ""),
            "correctness": row.get("correctness", ""),
        }
        for row in rows
    ]


def blocking_gaps(checks: list[dict[str, Any]]) -> list[str]:
    return [check["why"] for check in checks if check["status"] != "pass"]
