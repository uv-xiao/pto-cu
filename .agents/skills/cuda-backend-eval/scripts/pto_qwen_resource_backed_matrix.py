"""Matrix updates for PTO Qwen resource-backed viewer imports."""

from __future__ import annotations

from typing import Any


COVERAGE = "diagnostic_resource_backed_qwen_dag"


def ensure_matrix_ref(
    matrix: dict[str, Any],
    *,
    raw_artifact: str | None = None,
) -> dict[str, Any]:
    updated = dict(matrix)
    records = []
    for record in matrix["paper_evaluation_matrix"]:
        current = dict(record)
        if current["id"] == "llm_serving_paper_baselines":
            ensure_claim_refs(current, raw_artifact=raw_artifact)
            update_pto_gap_action(current)
        records.append(current)
    updated["paper_evaluation_matrix"] = records
    return updated


def ensure_claim_refs(record: dict[str, Any], *, raw_artifact: str | None) -> None:
    refs = record["current_evidence_refs"]
    viewer_ref = {
        "kind": "viewer_result",
        "benchmark_id": "llm_serving_decode",
        "method_id": "pto_persistent_device",
        "gpu": "A100",
        "shape_contains": "Qwen/Qwen3-8B resource-backed diagnostic",
        "serving_coverage": COVERAGE,
    }
    if viewer_ref not in refs:
        refs.append(viewer_ref)
    if raw_artifact is not None:
        refs[:] = [
            item
            for item in refs
            if not (
                item.get("kind") == "raw_artifact"
                and item.get("path") == raw_artifact
            )
        ]
        refs.append(raw_artifact_ref(raw_artifact))


def raw_artifact_ref(raw_artifact: str) -> dict[str, Any]:
    return {
        "kind": "raw_artifact",
        "path": raw_artifact,
        "symbols": [
            "pto_qwen_resource_backed_execution",
            "qwen_resource_backed_diagnostic_execution",
            "qwen_resource_backed_decode_step_execution",
            "qwen_resource_backed_policy_length_decode_execution",
            "qwen_diagnostic_decode_token_feedback",
            "qwen_device_decode_token_feedback",
            "qwen_resource_backed_unit_numeric_task_mode",
            "qwen_resource_backed_external_rmsnorm_scale",
            "qwen_resource_backed_full_rmsnorm_reduction",
            "qwen_resource_backed_weighted_elementwise_branches",
            "diagnostic_resource_backed_qwen_dag",
            "repeat_runs",
            "partial_logits_not_full_vocab",
            "full_logits_buffer_prefix_sampled",
            "full_logits_buffer_checked",
            "final_step_logits_check_policy",
            "diagnostic_qwen_logits_formula",
            "logits_summary_stable",
        ],
    }


def update_pto_gap_action(record: dict[str, Any]) -> None:
    for detail in record.get("missing_evidence_details", []):
        if detail.get("id") != "pto_full_serving_qwen3_8b":
            continue
        action = detail["action"]
        for phrase in old_action_phrases():
            action = action.replace(phrase, current_action_phrase())
        detail["action"] = action
        detail["evidence_summary"] = [
            summary.replace(
                "bounded full-RMSNorm reduction diagnostic execution",
                "block-wide full-vector RMSNorm reduction diagnostic execution",
            )
            for summary in detail.get("evidence_summary", [])
        ]


def current_action_phrase() -> str:
    return (
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports with full-logits-buffer diagnostic writes and "
        "bounded-prefix diagnostic reference checks plus device-side "
        "diagnostic sampled-token feedback, final-step logits-check policy, "
        "block-wide full-vector RMSNorm reduction diagnostics, and policy-length "
        "MPK/VDCores diagnostic decode runs are present. Remaining gates: "
        "full Qwen numerical correctness and full-serving viewer_result_import."
    )


def old_action_phrases() -> tuple[str, ...]:
    return (
        "diagnostic proxy, unit-math, and descriptor-smoke "
        "viewer_result_imports are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "and resource-backed execution viewer_result_imports are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports with partial-logits sampling are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports with full-logits-buffer diagnostic writes and "
        "bounded-prefix sampling are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports with full-logits-buffer diagnostic writes and "
        "bounded-prefix diagnostic reference checks are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports with full-logits-buffer diagnostic writes and "
        "bounded-prefix diagnostic reference checks plus device-side "
        "diagnostic sampled-token feedback are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports with full-logits-buffer diagnostic writes and "
        "bounded-prefix diagnostic reference checks plus device-side "
        "diagnostic sampled-token feedback and final-step logits-check "
        "policy are present.",
        "diagnostic proxy, unit-math, descriptor-smoke, "
        "resource-backed execution, and repeated resource-backed execution "
        "viewer_result_imports with full-logits-buffer diagnostic writes and "
        "bounded-prefix diagnostic reference checks plus device-side "
        "diagnostic sampled-token feedback, final-step logits-check policy, "
        "bounded full-RMSNorm reduction diagnostics, and policy-length "
        "MPK/VDCores diagnostic decode runs are present.",
    )
