from __future__ import annotations


COLLECTION_KEYS = (
    "capture_imports",
    "claim_audits",
    "result_records",
    "paper_baseline_execution_attempts",
    "paper_baseline_environment_attempts",
    "paper_baseline_probes",
    "paper_baseline_runs",
    "paper_baseline_run_readiness",
    "paper_evaluation_matrix",
    "serving_command_plans",
)
SIDECAR_LIST_FIELDS = {
    "claim_audits": (
        "missing_evidence_details",
        "paper_baseline_run_statuses",
        "paper_baseline_run_readiness_statuses",
        "execution_attempt_statuses",
        "probe_statuses",
        "next_actions",
    ),
    "paper_evaluation_matrix": (
        "current_evidence_refs",
        "missing_evidence_details",
    ),
}
