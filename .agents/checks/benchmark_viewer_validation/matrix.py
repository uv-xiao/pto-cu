from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403


def validate_paper_evaluation_matrix(
    data: dict[str, Any],
    benchmark_ids: set[str],
    method_ids: set[str],
    baseline_ids: set[str],
    serving_workload_ids: set[str],
    results: dict[str, Any],
    root: Path,
) -> set[str]:
    records = require_list(
        data, "paper_evaluation_matrix", "paper evaluation matrix"
    )
    matrix_ids = check_unique_ids(records, "paper evaluation matrix")
    required_claims = {
        "host_schedule_launch_overhead",
        "persistent_device_scheduler_overhead",
        "tensor_core_tile_baselines",
        "llm_serving_paper_baselines",
    }
    if not required_claims <= matrix_ids:
        missing = sorted(required_claims - matrix_ids)
        fail(f"missing paper evaluation matrix claims: {missing}")

    result_index = [
        (
            result["benchmark_id"],
            result["method_id"],
            result["hardware"]["gpu"],
            result.get("inputs", {}).get("shape", ""),
            result.get("statistic", {}).get("serving_coverage", ""),
            result_workload_id(result),
        )
        for result in results["result_records"]
    ]
    baseline_coverage: set[str] = set()
    method_coverage: set[str] = set()
    hardware_coverage: set[str] = set()
    allowed_status = {
        "planned_no_results",
        "partial_current_capture",
        "ready_for_paper_claim",
    }
    required_metrics = {"correctness", "raw_artifacts"}

    for record in records:
        owner = f"paper evaluation matrix {record['id']}"
        for key in ("title", "claim", "status", "promotion_gate"):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")

        workloads = require_list(record, "workload_ids", owner)
        methods = require_list(record, "method_ids", owner)
        baselines = record.get("paper_baseline_ids", [])
        if not isinstance(baselines, list):
            fail(f"{owner} paper_baseline_ids is not a list")
        hardware_targets = require_list(record, "hardware_targets", owner)
        metrics = set(require_list(record, "required_metrics", owner))
        evidence_refs = require_list(record, "current_evidence_refs", owner)
        missing_evidence = record.get("missing_evidence")
        if not isinstance(missing_evidence, list) or not all(
            isinstance(item, str) and item.strip() for item in missing_evidence
        ):
            fail(f"{owner} missing_evidence is not a list of strings")
        if record["status"] == "ready_for_paper_claim" and missing_evidence:
            fail(f"{owner} is ready but still has missing_evidence")
        if record["status"] != "ready_for_paper_claim" and not missing_evidence:
            fail(f"{owner} is not ready but has no missing_evidence")
        exceptions = record.get("evidence_policy_exceptions", [])
        if not isinstance(exceptions, list):
            fail(f"{owner} evidence_policy_exceptions is not a list")
        if record["status"] == "ready_for_paper_claim":
            for exception in exceptions:
                if not isinstance(exception, dict):
                    fail(
                        f"{owner} evidence_policy_exceptions item is not an "
                        "object"
                    )
                for key in (
                    "id",
                    "title",
                    "status",
                    "scope",
                    "decision",
                    "rationale",
                    "review_rule",
                ):
                    require_string(exception, key, owner)
                if exception["status"] != "accepted":
                    fail(
                        f"{owner} ready exception must be accepted: "
                        f"{exception['id']}"
                    )
                validate_policy_exception_refs(
                    require_list(exception, "evidence_refs", owner),
                    owner,
                    root,
                )
        missing_details = record.get("missing_evidence_details", [])
        if not isinstance(missing_details, list):
            fail(f"{owner} missing_evidence_details is not a list")
        for detail in missing_details:
            if not isinstance(detail, dict):
                fail(f"{owner} missing_evidence_details item is not an object")
            for key in ("id", "status", "action"):
                require_string(detail, key, owner)
            method_id = detail.get("method_id", "")
            if not isinstance(method_id, str):
                fail(f"{owner} missing detail method_id is not a string")
            if method_id and method_id not in method_ids:
                fail(
                    f"{owner} missing detail references unknown "
                    f"method_id: {method_id}"
                )
            baseline_id = detail.get("paper_baseline_id", "")
            if not isinstance(baseline_id, str):
                fail(f"{owner} missing detail paper_baseline_id is not a string")
            if baseline_id and baseline_id not in baseline_ids:
                fail(
                    f"{owner} missing detail references unknown "
                    f"paper_baseline_id: {baseline_id}"
                )
            run_id = detail.get("paper_baseline_run_id", "")
            if not isinstance(run_id, str):
                fail(f"{owner} missing detail paper_baseline_run_id is not a string")
            shape_contains = detail.get("shape_contains", "")
            if not isinstance(shape_contains, str):
                fail(f"{owner} missing detail shape_contains is not a string")
            serving_ids = detail.get("serving_workload_ids")
            if not isinstance(serving_ids, list) or not all(
                isinstance(serving_id, str) and serving_id
                for serving_id in serving_ids
            ):
                fail(f"{owner} missing detail serving_workload_ids is invalid")
            evidence_summary = detail.get("evidence_summary", [])
            if not isinstance(evidence_summary, list) or not all(
                isinstance(item, str) and item.strip()
                for item in evidence_summary
            ):
                fail(f"{owner} missing detail evidence_summary is invalid")
            for serving_id in serving_ids:
                if serving_id not in serving_workload_ids:
                    fail(
                        f"{owner} missing detail references unknown "
                        f"serving_workload_id: {serving_id}"
                    )

        for workload_id in workloads:
            if workload_id not in benchmark_ids:
                fail(f"{owner} references unknown workload_id: {workload_id}")
        for method_id in methods:
            if method_id not in method_ids:
                fail(f"{owner} references unknown method_id: {method_id}")
            method_coverage.add(method_id)
        for baseline_id in baselines:
            if baseline_id not in baseline_ids:
                fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
            baseline_coverage.add(baseline_id)
        for hardware in hardware_targets:
            if not isinstance(hardware, str) or not hardware:
                fail(f"{owner} has invalid hardware target")
            hardware_coverage.add(hardware)
        for metric in metrics:
            if not isinstance(metric, str) or not metric:
                fail(f"{owner} has invalid required metric")
        if not required_metrics <= metrics:
            missing = sorted(required_metrics - metrics)
            fail(f"{owner} missing required metrics: {missing}")

        for ref in evidence_refs:
            if not isinstance(ref, dict):
                fail(f"{owner} current evidence ref is not an object")
            kind = require_string(ref, "kind", owner)
            if kind == "viewer_result":
                key = (
                    require_string(ref, "benchmark_id", owner),
                    require_string(ref, "method_id", owner),
                    require_string(ref, "gpu", owner),
                )
                shape_contains = ref.get("shape_contains")
                if shape_contains is not None and (
                    not isinstance(shape_contains, str) or not shape_contains
                ):
                    fail(f"{owner} viewer_result shape_contains is invalid")
                serving_coverage = ref.get("serving_coverage")
                if key[0] == "llm_serving_decode":
                    if not isinstance(serving_coverage, str) or not serving_coverage:
                        fail(
                            f"{owner} llm_serving_decode viewer_result lacks "
                            "serving_coverage"
                        )
                elif serving_coverage is not None:
                    fail(f"{owner} non-serving viewer_result has serving_coverage")
                required_workload_ids = ref.get("required_workload_ids", [])
                if not isinstance(required_workload_ids, list) or not all(
                    isinstance(workload_id, str)
                    and workload_id in serving_workload_ids
                    for workload_id in required_workload_ids
                ):
                    fail(f"{owner} viewer_result required_workload_ids is invalid")
                if required_workload_ids and not has_required_workload_results(
                    result_index,
                    key=key,
                    shape_contains=shape_contains,
                    serving_coverage=serving_coverage,
                    required_workload_ids=required_workload_ids,
                ):
                    fail(
                        f"{owner} viewer_result evidence lacks required workloads"
                    )
                if not any(
                    result_key[:3] == key
                    and (
                        shape_contains is None
                        or shape_contains in str(result_key[3])
                    )
                    and (
                        serving_coverage is None
                        or serving_coverage == result_key[4]
                    )
                    for result_key in result_index
                ):
                    detail = key if shape_contains is None else (*key, shape_contains)
                    fail(f"{owner} viewer_result evidence is missing: {detail}")
            elif kind in {
                "viewer_data",
                "stable_doc",
                "baseline_survey",
            }:
                path = require_string(ref, "path", owner)
                if not logical_data_path_exists(root, path):
                    fail(f"{owner} evidence path missing: {path}")
            elif kind == "raw_artifact":
                path = require_string(ref, "path", owner)
                require_current_artifact_path(root, path, owner)
            else:
                fail(f"{owner} has unknown evidence kind: {kind}")

    required_baselines = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    if not required_baselines <= baseline_coverage:
        missing = sorted(required_baselines - baseline_coverage)
        fail(f"paper evaluation matrix missing baseline coverage: {missing}")
    required_methods = {"pto_host_schedule", "pto_persistent_device"}
    if not required_methods <= method_coverage:
        missing = sorted(required_methods - method_coverage)
        fail(f"paper evaluation matrix missing PTO method coverage: {missing}")
    if not {"A100", "H200"} <= hardware_coverage:
        fail("paper evaluation matrix must cover A100 and H200")
    return matrix_ids


def result_workload_id(result: dict[str, Any]) -> str:
    statistic = result.get("statistic", {})
    if not isinstance(statistic, dict):
        statistic = {}
    workload_id = statistic.get("workload_id")
    if isinstance(workload_id, str) and workload_id:
        return workload_id
    shape = str(result.get("inputs", {}).get("shape", ""))
    for candidate in ("mpk_offline_decode", "vdcores_offline_decode"):
        if candidate in shape:
            return candidate
    return ""


def has_required_workload_results(
    result_index: list[tuple[str, str, str, Any, Any, str]],
    *,
    key: tuple[str, str, str],
    shape_contains: Any,
    serving_coverage: Any,
    required_workload_ids: list[str],
) -> bool:
    remaining = set(required_workload_ids)
    for result_key in result_index:
        if result_key[:3] != key:
            continue
        if shape_contains is not None and shape_contains not in str(result_key[3]):
            continue
        if serving_coverage is not None and serving_coverage != result_key[4]:
            continue
        remaining.discard(result_key[5])
    return not remaining
