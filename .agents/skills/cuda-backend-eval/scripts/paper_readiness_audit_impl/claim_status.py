from __future__ import annotations

from collections import Counter
from typing import Any

from .errors import fail


ResultKey = tuple[str, str, str, str, str, str, str, bool]
PTO_FULL_SERVING_WORKLOAD_IDS = {"mpk_offline_decode", "vdcores_offline_decode"}
PTO_FULL_SERVING_METRIC_FIELDS = {
    "batch_size",
    "decode_tokens",
    "end_to_end_latency_ns",
    "inter_token_latency_ns",
    "throughput_tokens_per_s",
    "time_to_first_token_ns",
}


def result_index(results: dict[str, Any]) -> set[ResultKey]:
    records = require_records(results, "result_records")
    index: set[ResultKey] = set()
    for record in records:
        hardware = record.get("hardware")
        inputs = record.get("inputs")
        statistic = record.get("statistic")
        if not isinstance(hardware, dict):
            continue
        if not isinstance(inputs, dict):
            inputs = {}
        if not isinstance(statistic, dict):
            statistic = {}
        benchmark_id = record.get("benchmark_id")
        method_id = record.get("method_id")
        gpu = hardware.get("gpu")
        shape = inputs.get("shape", "")
        serving_coverage = statistic.get("serving_coverage", "")
        workload_id = pto_workload_id(str(shape), statistic)
        pto_full_serving_ready = pto_full_serving_row_ready(record, statistic)
        if all(isinstance(value, str) for value in (benchmark_id, method_id, gpu)):
            index.add(
                (
                    benchmark_id,
                    method_id,
                    gpu,
                    str(shape),
                    str(serving_coverage),
                    str(record.get("correctness", "")),
                    workload_id,
                    pto_full_serving_ready,
                )
            )
    return index


def pto_workload_id(shape: str, statistic: dict[str, Any]) -> str:
    workload_id = statistic.get("workload_id")
    if isinstance(workload_id, str) and workload_id:
        return workload_id
    for candidate in PTO_FULL_SERVING_WORKLOAD_IDS:
        if candidate in shape:
            return candidate
    return ""


def pto_full_serving_row_ready(
    record: dict[str, Any],
    statistic: dict[str, Any],
) -> bool:
    shape = str(record.get("inputs", {}).get("shape", ""))
    if record.get("benchmark_id") != "llm_serving_decode":
        return False
    if record.get("method_id") != "pto_persistent_device":
        return False
    if "Qwen/Qwen3-8B" not in shape:
        return False
    if statistic.get("serving_coverage") != "full_serving":
        return False
    if pto_workload_id(shape, statistic) not in PTO_FULL_SERVING_WORKLOAD_IDS:
        return False
    if record.get("correctness") != "pass":
        return False
    if not record.get("raw_artifact"):
        return False
    for key in PTO_FULL_SERVING_METRIC_FIELDS:
        value = statistic.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            return False
    return True


def require_records(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    records = data.get(key)
    if not isinstance(records, list):
        fail(f"{key} is missing or not a list")
    if not all(isinstance(record, dict) for record in records):
        fail(f"{key} contains a non-object record")
    return records


def has_viewer_result(
    current_results: set[ResultKey],
    ref: dict[str, Any],
) -> bool:
    benchmark_id = str(ref.get("benchmark_id", ""))
    method_id = str(ref.get("method_id", ""))
    gpu = str(ref.get("gpu", ""))
    shape_contains = ref.get("shape_contains")
    serving_coverage = ref.get("serving_coverage")
    required_workload_ids = ref.get("required_workload_ids", [])
    if required_workload_ids:
        return has_required_viewer_results(
            current_results,
            benchmark_id=benchmark_id,
            method_id=method_id,
            gpu=gpu,
            shape_contains=shape_contains,
            serving_coverage=serving_coverage,
            required_workload_ids=required_workload_ids,
        )
    for (
        current_benchmark,
        current_method,
        current_gpu,
        current_shape,
        current_coverage,
        _current_correctness,
        _current_workload_id,
        current_pto_full_serving_ready,
    ) in current_results:
        if (
            current_benchmark,
            current_method,
            current_gpu,
        ) != (benchmark_id, method_id, gpu):
            continue
        if (
            isinstance(serving_coverage, str)
            and serving_coverage
            and serving_coverage != current_coverage
        ):
            continue
        if (
            benchmark_id == "llm_serving_decode"
            and method_id == "pto_persistent_device"
            and serving_coverage == "full_serving"
            and not current_pto_full_serving_ready
        ):
            continue
        if shape_contains is None:
            return True
        if isinstance(shape_contains, str) and shape_contains in current_shape:
            return True
    return False


def has_required_viewer_results(
    current_results: set[ResultKey],
    *,
    benchmark_id: str,
    method_id: str,
    gpu: str,
    shape_contains: Any,
    serving_coverage: Any,
    required_workload_ids: Any,
) -> bool:
    if not isinstance(required_workload_ids, list) or not all(
        isinstance(workload_id, str) and workload_id
        for workload_id in required_workload_ids
    ):
        return False
    remaining = set(required_workload_ids)
    for (
        current_benchmark,
        current_method,
        current_gpu,
        current_shape,
        current_coverage,
        _current_correctness,
        current_workload_id,
        current_pto_full_serving_ready,
    ) in current_results:
        if (
            current_benchmark,
            current_method,
            current_gpu,
        ) != (benchmark_id, method_id, gpu):
            continue
        if (
            isinstance(serving_coverage, str)
            and serving_coverage
            and serving_coverage != current_coverage
        ):
            continue
        if (
            benchmark_id == "llm_serving_decode"
            and method_id == "pto_persistent_device"
            and serving_coverage == "full_serving"
            and not current_pto_full_serving_ready
        ):
            continue
        if isinstance(shape_contains, str) and shape_contains not in current_shape:
            continue
        remaining.discard(current_workload_id)
    return not remaining


def count_evidence_refs(
    evidence_refs: list[dict[str, Any]],
    current_results: set[ResultKey],
) -> tuple[dict[str, int], list[str]]:
    counts = Counter()
    missing_viewer_results: list[str] = []
    for ref in evidence_refs:
        kind = ref.get("kind")
        if not isinstance(kind, str):
            kind = "unknown"
        counts[kind] += 1
        if kind == "viewer_result":
            key = [
                str(ref.get("benchmark_id", "")),
                str(ref.get("method_id", "")),
                str(ref.get("gpu", "")),
            ]
            shape_contains = ref.get("shape_contains")
            if isinstance(shape_contains, str) and shape_contains:
                key.append(f"shape contains {shape_contains}")
            serving_coverage = ref.get("serving_coverage")
            if isinstance(serving_coverage, str) and serving_coverage:
                key.append(f"coverage {serving_coverage}")
            required_workload_ids = ref.get("required_workload_ids")
            if isinstance(required_workload_ids, list) and required_workload_ids:
                key.append("workloads " + ",".join(map(str, required_workload_ids)))
            if not has_viewer_result(current_results, ref):
                missing_viewer_results.append(" / ".join(key))
    return dict(sorted(counts.items())), missing_viewer_results


def paper_baseline_run_statuses(
    claim_id: str,
    paper_baseline_ids: list[str],
    runs_by_claim: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    baseline_filter = set(paper_baseline_ids)
    statuses: list[dict[str, Any]] = []
    for run in runs_by_claim.get(claim_id, []):
        baseline_id = run.get("paper_baseline_id")
        if baseline_filter and baseline_id not in baseline_filter:
            continue
        statuses.append(
            {
                "id": run["id"],
                "paper_baseline_id": baseline_id,
                "status": run["status"],
                "serving_workload_ids": run.get("serving_workload_ids", []),
                "expected_artifacts": run.get("expected_artifacts", []),
            }
        )
    return statuses


def probe_statuses(
    paper_baseline_ids: list[str],
    probes_by_baseline: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for baseline_id in paper_baseline_ids:
        probe = probes_by_baseline.get(baseline_id)
        if probe is None:
            statuses.append(
                {
                    "paper_baseline_id": baseline_id,
                    "latest_status": "missing_probe_record",
                    "machines": [],
                    "next_action": (
                        "Add a paper-baseline probe record before executing "
                        "or importing this baseline."
                    ),
                }
            )
            continue
        statuses.append(
            {
                "paper_baseline_id": baseline_id,
                "latest_status": probe["latest_status"],
                "next_action": probe["next_action"],
                "machines": [
                    {
                        "gpu": machine["gpu"],
                        "status": machine["status"],
                        "blocking_gaps": machine.get("blocking_gaps", []),
                    }
                    for machine in probe.get("latest_machine_status", [])
                ],
            }
        )
    return statuses


def paper_baseline_run_readiness_statuses(
    run_statuses: list[dict[str, Any]],
    readiness_by_run: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for run in run_statuses:
        if run["status"] == "imported_to_viewer":
            continue
        run_id = run["id"]
        readiness = readiness_by_run.get(run_id)
        if readiness is None:
            statuses.append(
                {
                    "paper_baseline_run_id": run_id,
                    "paper_baseline_id": run["paper_baseline_id"],
                    "latest_status": "missing_readiness_record",
                    "latest_artifact_root": "",
                    "blocking_gaps": [
                        f"No paper-baseline run-readiness record exists for {run_id}."
                    ],
                    "next_action": (
                        "Add a paper-baseline run-readiness record before "
                        "executing or importing this run."
                    ),
                }
            )
            continue
        statuses.append(
            {
                "paper_baseline_run_id": run_id,
                "paper_baseline_id": readiness["paper_baseline_id"],
                "latest_status": readiness["latest_status"],
                "latest_artifact_root": readiness["latest_artifact_root"],
                "blocking_gaps": readiness.get("blocking_gaps", []),
                "next_action": readiness["next_action"],
            }
        )
    return statuses


def execution_attempt_statuses(
    run_statuses: list[dict[str, Any]],
    attempts_by_run: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for run in run_statuses:
        if run["status"] == "imported_to_viewer":
            continue
        run_id = run["id"]
        attempts = attempts_by_run.get(run_id, [])
        if not attempts:
            continue
        latest = attempts[-1]
        summary = latest.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        statuses.append(
            {
                "paper_baseline_run_id": run_id,
                "paper_baseline_id": latest["paper_baseline_id"],
                "execution_attempt_id": latest["id"],
                "title": latest["title"],
                "status": latest["status"],
                "artifact_root": latest["artifact_root"],
                "blocker": latest.get("blocker", ""),
                "summary": summary,
            }
        )
    return statuses


def claim_blockers(
    *,
    paper_baseline_ids: list[str],
    missing_evidence: list[str],
    missing_viewer_results: list[str],
    run_statuses: list[dict[str, Any]],
    run_readiness_statuses: list[dict[str, Any]],
    execution_attempts: list[dict[str, Any]],
    probes: list[dict[str, Any]],
) -> list[str]:
    blockers = list(missing_evidence)
    for result in missing_viewer_results:
        blockers.append(f"Missing current viewer result evidence: {result}.")
    for run in run_statuses:
        if run["status"] != "imported_to_viewer":
            blockers.append(
                f"Paper baseline run {run['id']} is {run['status']}, not imported_to_viewer."
            )
    covered_run_baselines = {run["paper_baseline_id"] for run in run_statuses}
    for baseline_id in paper_baseline_ids:
        if baseline_id not in covered_run_baselines:
            blockers.append(
                f"No paper baseline run record is attached to this claim for {baseline_id}."
            )
    for probe in probes:
        if probe["latest_status"] != "pass":
            blockers.append(
                f"Readiness probe for {probe['paper_baseline_id']} is {probe['latest_status']}."
            )
    for readiness in run_readiness_statuses:
        if readiness["latest_status"] == "pass":
            continue
        gaps = readiness.get("blocking_gaps", [])
        gap_text = "; ".join(gap.rstrip(".") for gap in gaps)
        detail = f": {gap_text}." if gap_text else "."
        blockers.append(
            "Run readiness "
            f"{readiness['paper_baseline_run_id']} is "
            f"{readiness['latest_status']}{detail}"
        )
    for attempt in execution_attempts:
        if attempt["status"] == "pass":
            continue
        blocker = str(attempt.get("blocker", "")).strip()
        detail = f": {blocker}" if blocker else ""
        blockers.append(
            "Latest execution attempt "
            f"{attempt['execution_attempt_id']} for "
            f"{attempt['paper_baseline_run_id']} is "
            f"{attempt['status']}{detail}"
        )
    return blockers
