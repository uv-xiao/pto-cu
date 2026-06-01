from __future__ import annotations

from typing import Any


def claim_next_actions(
    *,
    missing_evidence: list[str],
    missing_evidence_details: list[dict[str, Any]],
    run_readiness_statuses: list[dict[str, Any]],
    execution_attempts: list[dict[str, Any]],
    probes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()

    def append_action(action: dict[str, Any]) -> None:
        key = (
            str(action.get("source", "")),
            str(action.get("missing_evidence_id", "")),
            str(action.get("paper_baseline_run_id", "")),
            str(action.get("paper_baseline_id", "")),
            str(action.get("method_id", "")),
            str(action.get("shape_contains", "")),
        )
        if key in seen:
            return
        seen.add(key)
        actions.append(action)

    if missing_evidence_details:
        for detail in missing_evidence_details:
            action = detail.get("action", "")
            append_action(
                {
                    "source": "matrix_missing_evidence",
                    "missing_evidence_id": str(detail.get("id", "")),
                    "paper_baseline_id": str(detail.get("paper_baseline_id", "")),
                    "paper_baseline_run_id": str(
                        detail.get("paper_baseline_run_id", "")
                    ),
                    "method_id": str(detail.get("method_id", "")),
                    "serving_workload_ids": detail.get("serving_workload_ids", []),
                    "shape_contains": str(detail.get("shape_contains", "")),
                    "status": str(detail.get("status", "missing")),
                    "action": str(action),
                    "evidence_summary": detail.get("evidence_summary", []),
                }
            )
    for item in [] if missing_evidence_details else missing_evidence:
        append_action(
            {
                "source": "matrix_missing_evidence",
                "status": "missing",
                "action": item,
            }
        )
    for readiness in run_readiness_statuses:
        if readiness["latest_status"] == "pass":
            continue
        action = readiness.get("next_action")
        if not isinstance(action, str) or not action.strip():
            continue
        append_action(
            {
                "source": "run_readiness",
                "paper_baseline_id": readiness["paper_baseline_id"],
                "paper_baseline_run_id": readiness["paper_baseline_run_id"],
                "status": readiness["latest_status"],
                "action": action,
            }
        )
    for attempt in execution_attempts:
        if attempt["status"] == "pass":
            continue
        blocker = str(attempt.get("blocker", "")).strip()
        action = (
            f"Resolve latest execution attempt {attempt['execution_attempt_id']}"
            f" before importing {attempt['paper_baseline_run_id']}."
        )
        if blocker:
            action = f"{action} Diagnostic blocker: {blocker}"
        append_action(
            {
                "source": "execution_attempt",
                "paper_baseline_id": attempt["paper_baseline_id"],
                "paper_baseline_run_id": attempt["paper_baseline_run_id"],
                "execution_attempt_id": attempt["execution_attempt_id"],
                "status": attempt["status"],
                "action": action,
            }
        )
    for probe in probes:
        if probe["latest_status"] == "pass":
            continue
        action = probe.get("next_action")
        if not isinstance(action, str) or not action.strip():
            continue
        append_action(
            {
                "source": "probe",
                "paper_baseline_id": probe["paper_baseline_id"],
                "status": probe["latest_status"],
                "action": action,
            }
        )
    return actions
