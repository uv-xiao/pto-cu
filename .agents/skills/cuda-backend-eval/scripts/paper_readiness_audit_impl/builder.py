from __future__ import annotations

from collections import defaultdict
from typing import Any

from .claim_actions import claim_next_actions
from .claim_status import claim_blockers
from .claim_status import count_evidence_refs
from .claim_status import execution_attempt_statuses
from .claim_status import paper_baseline_run_readiness_statuses
from .claim_status import paper_baseline_run_statuses
from .claim_status import probe_statuses
from .claim_status import require_records
from .claim_status import result_index
from .errors import fail


def validate_missing_evidence_details(
    claim: dict[str, Any],
    missing_evidence_details: list[Any],
) -> None:
    if not isinstance(missing_evidence_details, list):
        fail(f"{claim.get('id', '<unknown>')} missing_evidence_details is invalid")
    for detail in missing_evidence_details:
        if not isinstance(detail, dict):
            fail(
                f"{claim.get('id', '<unknown>')} "
                "missing_evidence_details contains a non-object"
            )
        for key in ("id", "status", "action"):
            value = detail.get(key)
            if not isinstance(value, str) or not value.strip():
                fail(
                    f"{claim.get('id', '<unknown>')} "
                    f"missing_evidence_details has invalid {key}"
                )
        serving_ids = detail.get("serving_workload_ids", [])
        if not isinstance(serving_ids, list) or not all(
            isinstance(item, str) and item for item in serving_ids
        ):
            fail(
                f"{claim.get('id', '<unknown>')} "
                "missing_evidence_details has invalid serving_workload_ids"
            )
        for key in (
            "paper_baseline_id",
            "paper_baseline_run_id",
            "method_id",
            "shape_contains",
        ):
            value = detail.get(key, "")
            if not isinstance(value, str):
                fail(
                    f"{claim.get('id', '<unknown>')} "
                    f"missing_evidence_details has invalid {key}"
                )
        evidence_summary = detail.get("evidence_summary", [])
        if not isinstance(evidence_summary, list) or not all(
            isinstance(item, str) and item.strip() for item in evidence_summary
        ):
            fail(
                f"{claim.get('id', '<unknown>')} "
                "missing_evidence_details has invalid evidence_summary"
            )


def build_readiness_audit(
    *,
    matrix: dict[str, Any],
    runs: dict[str, Any],
    probes: dict[str, Any],
    run_readiness: dict[str, Any],
    execution_attempts: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    claims = require_records(matrix, "paper_evaluation_matrix")
    run_records = require_records(runs, "paper_baseline_runs")
    probe_records = require_records(probes, "paper_baseline_probes")
    readiness_records = require_records(
        run_readiness,
        "paper_baseline_run_readiness",
    )
    attempt_records = require_records(
        execution_attempts,
        "paper_baseline_execution_attempts",
    )
    current_results = result_index(results)

    runs_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in run_records:
        paper_evaluation_id = run.get("paper_evaluation_id")
        if isinstance(paper_evaluation_id, str):
            runs_by_claim[paper_evaluation_id].append(run)
    probes_by_baseline = {
        probe["paper_baseline_id"]: probe
        for probe in probe_records
        if isinstance(probe.get("paper_baseline_id"), str)
    }
    readiness_by_run = {
        readiness["paper_baseline_run_id"]: readiness
        for readiness in readiness_records
        if isinstance(readiness.get("paper_baseline_run_id"), str)
    }
    attempts_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempt_records:
        paper_baseline_run_id = attempt.get("paper_baseline_run_id")
        if isinstance(paper_baseline_run_id, str):
            attempts_by_run[paper_baseline_run_id].append(attempt)

    claim_audits: list[dict[str, Any]] = []
    for claim in claims:
        evidence_refs = claim.get("current_evidence_refs", [])
        if not isinstance(evidence_refs, list):
            fail(f"{claim.get('id', '<unknown>')} current_evidence_refs is not a list")
        evidence_counts, missing_results = count_evidence_refs(
            evidence_refs,
            current_results,
        )
        missing_evidence = claim.get("missing_evidence", [])
        if not isinstance(missing_evidence, list) or not all(
            isinstance(item, str) for item in missing_evidence
        ):
            fail(f"{claim.get('id', '<unknown>')} missing_evidence is invalid")
        missing_evidence_details = claim.get("missing_evidence_details", [])
        validate_missing_evidence_details(claim, missing_evidence_details)
        evidence_policy_exceptions = claim.get("evidence_policy_exceptions", [])
        if not isinstance(evidence_policy_exceptions, list):
            fail(
                f"{claim.get('id', '<unknown>')} "
                "evidence_policy_exceptions is invalid"
            )
        paper_baseline_ids = claim.get("paper_baseline_ids", [])
        if not isinstance(paper_baseline_ids, list) or not all(
            isinstance(item, str) for item in paper_baseline_ids
        ):
            fail(f"{claim.get('id', '<unknown>')} paper_baseline_ids is invalid")

        run_statuses = paper_baseline_run_statuses(
            claim["id"],
            paper_baseline_ids,
            runs_by_claim,
        )
        probe_items = probe_statuses(paper_baseline_ids, probes_by_baseline)
        run_readiness_items = paper_baseline_run_readiness_statuses(
            run_statuses,
            readiness_by_run,
        )
        execution_attempt_items = execution_attempt_statuses(
            run_statuses,
            attempts_by_run,
        )
        blockers = claim_blockers(
            paper_baseline_ids=paper_baseline_ids,
            missing_evidence=missing_evidence,
            missing_viewer_results=missing_results,
            run_statuses=run_statuses,
            run_readiness_statuses=run_readiness_items,
            execution_attempts=execution_attempt_items,
            probes=probe_items,
        )
        next_actions = claim_next_actions(
            missing_evidence=missing_evidence,
            missing_evidence_details=missing_evidence_details,
            run_readiness_statuses=run_readiness_items,
            execution_attempts=execution_attempt_items,
            probes=probe_items,
        )
        ready = claim["status"] == "ready_for_paper_claim" and not blockers
        claim_audits.append(
            {
                "id": claim["id"],
                "title": claim["title"],
                "matrix_status": claim["status"],
                "ready_for_paper_claim": ready,
                "evidence_ref_counts": evidence_counts,
                "missing_evidence_count": len(missing_evidence),
                "missing_evidence_details": missing_evidence_details,
                "evidence_policy_exceptions": evidence_policy_exceptions,
                "missing_viewer_results": missing_results,
                "paper_baseline_run_statuses": run_statuses,
                "paper_baseline_run_readiness_statuses": run_readiness_items,
                "execution_attempt_statuses": execution_attempt_items,
                "probe_statuses": probe_items,
                "blockers": blockers,
                "next_actions": [] if ready else next_actions,
                "promotion_gate": claim["promotion_gate"],
            }
        )

    ready_claims = sum(1 for claim in claim_audits if claim["ready_for_paper_claim"])
    return {
        "schema_version": 1,
        "source_files": [
            "evaluations/nvidia/benchmark-viewer/data/paper_evaluation_matrix.json",
            "evaluations/nvidia/benchmark-viewer/data/paper_baseline_runs.json",
            "evaluations/nvidia/benchmark-viewer/data/paper_baseline_probes.json",
            "evaluations/nvidia/benchmark-viewer/data/paper_baseline_run_readiness.json",
            "evaluations/nvidia/benchmark-viewer/data/paper_baseline_execution_attempts.json",
            "evaluations/nvidia/benchmark-viewer/data/results.json",
        ],
        "overall_status": "paper_ready"
        if ready_claims == len(claim_audits)
        else "not_paper_ready",
        "ready_claims": ready_claims,
        "blocked_claims": len(claim_audits) - ready_claims,
        "claim_audits": claim_audits,
    }
