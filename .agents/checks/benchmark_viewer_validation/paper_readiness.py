from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403
from .generated_builders import (
    load_readiness_audit_builder,
    load_work_queue_builder,
)


def validate_paper_readiness_audit(
    audit: dict[str, Any],
    *,
    matrix: dict[str, Any],
    runs: dict[str, Any],
    probes: dict[str, Any],
    run_readiness: dict[str, Any],
    execution_attempts: dict[str, Any],
    results: dict[str, Any],
    serving_workload_ids: set[str],
) -> None:
    if audit.get("schema_version") != 1:
        fail("paper readiness audit schema_version must be 1")
    required_sources = {
        "evaluations/nvidia/benchmark-viewer/data/paper_evaluation_matrix.json",
        "evaluations/nvidia/benchmark-viewer/data/paper_baseline_runs.json",
        "evaluations/nvidia/benchmark-viewer/data/paper_baseline_probes.json",
        "evaluations/nvidia/benchmark-viewer/data/paper_baseline_run_readiness.json",
        "evaluations/nvidia/benchmark-viewer/data/paper_baseline_execution_attempts.json",
        "evaluations/nvidia/benchmark-viewer/data/results.json",
    }
    sources = audit.get("source_files")
    if not isinstance(sources, list) or set(sources) != required_sources:
        fail("paper readiness audit source_files are stale")
    claim_audits = audit.get("claim_audits")
    if not isinstance(claim_audits, list) or not claim_audits:
        fail("paper readiness audit has no claim_audits")
    for claim in claim_audits:
        if not isinstance(claim, dict):
            fail("paper readiness audit contains non-object claim")
        owner = f"paper readiness audit {claim.get('id', '<missing>')}"
        for key in (
            "id",
            "title",
            "matrix_status",
            "promotion_gate",
        ):
            require_string(claim, key, owner)
        if not isinstance(claim.get("ready_for_paper_claim"), bool):
            fail(f"{owner} ready_for_paper_claim is not boolean")
        if not isinstance(claim.get("evidence_ref_counts"), dict):
            fail(f"{owner} missing evidence_ref_counts")
        for key in (
            "missing_viewer_results",
            "paper_baseline_run_statuses",
            "paper_baseline_run_readiness_statuses",
            "execution_attempt_statuses",
            "probe_statuses",
            "blockers",
            "next_actions",
        ):
            value = claim.get(key)
            if not isinstance(value, list):
                fail(f"{owner} {key} is not a list")
        for action in claim["next_actions"]:
            if not isinstance(action, dict):
                fail(f"{owner} next action is not an object")
            source = require_string(action, "source", owner)
            if source not in {
                "matrix_missing_evidence",
                "run_readiness",
                "execution_attempt",
                "probe",
            }:
                fail(f"{owner} has invalid next action source: {source}")
            require_string(action, "status", owner)
            require_string(action, "action", owner)
            evidence_summary = action.get("evidence_summary", [])
            if not isinstance(evidence_summary, list) or not all(
                isinstance(item, str) and item.strip()
                for item in evidence_summary
            ):
                fail(f"{owner} next action evidence_summary is invalid")
            serving_ids = action.get("serving_workload_ids", [])
            if not isinstance(serving_ids, list) or not all(
                isinstance(item, str) and item for item in serving_ids
            ):
                fail(f"{owner} next action serving_workload_ids is invalid")
            for serving_id in serving_ids:
                if serving_id not in serving_workload_ids:
                    fail(
                        f"{owner} next action references unknown "
                        f"serving_workload_id: {serving_id}"
                    )
            for key in ("missing_evidence_id", "method_id", "shape_contains"):
                value = action.get(key, "")
                if not isinstance(value, str):
                    fail(f"{owner} next action {key} is not a string")
            if source in {"run_readiness", "execution_attempt", "probe"}:
                require_string(action, "paper_baseline_id", owner)
            if source in {"run_readiness", "execution_attempt"}:
                require_string(action, "paper_baseline_run_id", owner)
            if source == "execution_attempt":
                require_string(action, "execution_attempt_id", owner)
            if source == "matrix_missing_evidence":
                has_structured_target = any(
                    action.get(key)
                    for key in (
                        "missing_evidence_id",
                        "method_id",
                        "shape_contains",
                    )
                ) or bool(serving_ids)
                if has_structured_target:
                    require_string(action, "missing_evidence_id", owner)
                if serving_ids and not action.get("shape_contains"):
                    fail(f"{owner} matrix action with serving ids lacks shape")
        missing_count = claim.get("missing_evidence_count")
        if isinstance(missing_count, bool) or not isinstance(missing_count, int):
            fail(f"{owner} missing_evidence_count is not an integer")
        if claim["ready_for_paper_claim"] and (
            claim["blockers"] or missing_count != 0
        ):
            fail(f"{owner} is ready but still has blockers or missing evidence")
        if not claim["ready_for_paper_claim"] and not claim["blockers"]:
            fail(f"{owner} is blocked but has no blockers")

    generated = load_readiness_audit_builder()(
        matrix=matrix,
        runs=runs,
        probes=probes,
        run_readiness=run_readiness,
        execution_attempts=execution_attempts,
        results=results,
    )
    if audit != generated:
        fail("paper readiness audit is stale; regenerate paper_readiness_audit.json")


def validate_paper_readiness_work_queue(
    work_queue: dict[str, Any],
    *,
    audit: dict[str, Any],
    runs: dict[str, Any],
    serving_workload_ids: set[str],
    serving_command_plan: dict[str, Any],
) -> None:
    if work_queue.get("schema_version") != 1:
        fail("paper readiness work queue schema_version must be 1")
    if (
        work_queue.get("source_file")
        != "evaluations/nvidia/benchmark-viewer/data/paper_readiness_audit.json"
    ):
        fail("paper readiness work queue source_file is stale")
    if work_queue.get("overall_status") != audit.get("overall_status"):
        fail("paper readiness work queue overall_status does not match audit")
    if work_queue.get("ready_claims") != audit.get("ready_claims"):
        fail("paper readiness work queue ready_claims does not match audit")
    if work_queue.get("blocked_claims") != audit.get("blocked_claims"):
        fail("paper readiness work queue blocked_claims does not match audit")
    summary = work_queue.get("summary")
    if not isinstance(summary, dict):
        fail("paper readiness work queue has no summary object")
    work_items = work_queue.get("work_items")
    if not isinstance(work_items, list):
        fail("paper readiness work queue has no work_items list")
    if summary.get("total_work_items") != len(work_items):
        fail("paper readiness work queue summary total does not match items")
    expected_total = sum(
        len(claim.get("next_actions", []))
        for claim in audit.get("claim_audits", [])
    )
    if len(work_items) != expected_total:
        fail("paper readiness work queue item count does not match audit")
    item_ids: set[str] = set()
    command_plan_ids = {
        record["id"]
        for record in serving_command_plan.get("serving_command_plans", [])
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }
    run_serving_ids = {
        record["id"]: record.get("serving_workload_ids", [])
        for record in runs.get("paper_baseline_runs", [])
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }
    for expected_priority, item in enumerate(work_items, start=1):
        if not isinstance(item, dict):
            fail("paper readiness work queue item is not an object")
        owner = f"paper readiness work queue item {expected_priority}"
        item_id = require_string(item, "id", owner)
        validate_id(item_id, owner)
        if item_id in item_ids:
            fail(f"duplicate paper readiness work queue item id: {item_id}")
        item_ids.add(item_id)
        if item.get("priority") != expected_priority:
            fail(f"{owner} priority is not sequential")
        for key in (
            "claim_id",
            "claim_title",
            "matrix_status",
            "source",
            "owner",
            "status",
            "action",
            "promotion_gate",
        ):
            require_string(item, key, owner)
        evidence_summary = item.get("evidence_summary", [])
        if not isinstance(evidence_summary, list) or not all(
            isinstance(entry, str) and entry.strip()
            for entry in evidence_summary
        ):
            fail(f"{owner} evidence_summary is invalid")
        if not evidence_summary:
            fail(f"{owner} evidence_summary must not be empty")
        if not isinstance(item.get("ready_for_paper_claim"), bool):
            fail(f"{owner} ready_for_paper_claim is not boolean")
        for key in ("blocker_count", "missing_evidence_count", "action_index"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                fail(f"{owner} {key} is not an integer")
        for key in (
            "paper_baseline_id",
            "paper_baseline_run_id",
            "execution_attempt_id",
            "missing_evidence_id",
            "method_id",
            "shape_contains",
        ):
            if not isinstance(item.get(key), str):
                fail(f"{owner} {key} is not a string")
        serving_ids = item.get("serving_workload_ids")
        if not isinstance(serving_ids, list) or not all(
            isinstance(serving_id, str) and serving_id
            for serving_id in serving_ids
        ):
            fail(f"{owner} serving_workload_ids is not a string list")
        for serving_id in serving_ids:
            if serving_id not in serving_workload_ids:
                fail(
                    f"{owner} references unknown serving_workload_id: "
                    f"{serving_id}"
                )
        command_selectors = item.get("serving_command_plan_selectors")
        if not isinstance(command_selectors, list) or not all(
            isinstance(selector, str) and selector
            for selector in command_selectors
        ):
            fail(f"{owner} serving_command_plan_selectors is not a string list")
        expected_serving_ids = serving_ids
        if item["paper_baseline_run_id"] in run_serving_ids:
            allowed_serving_ids = set(run_serving_ids[item["paper_baseline_run_id"]])
            expected_serving_ids = [
                serving_id
                for serving_id in serving_ids
                if serving_id in allowed_serving_ids
            ]
        expected_selectors = [
            f"{item['paper_baseline_run_id']}:{serving_id}"
            for serving_id in expected_serving_ids
            if item["paper_baseline_run_id"]
        ]
        if command_selectors != expected_selectors:
            fail(f"{owner} serving command selectors are stale")
        for selector in command_selectors:
            if not any(
                plan_id.startswith(f"{selector}:batch")
                for plan_id in command_plan_ids
            ):
                fail(f"{owner} selector has no serving command plan: {selector}")
        if item["ready_for_paper_claim"]:
            fail(f"{owner} points at a ready paper claim")
    generated = load_work_queue_builder()(audit, runs=runs)
    if work_queue != generated:
        fail(
            "paper readiness work queue is stale; regenerate "
            "paper_readiness_work_queue.json"
        )
