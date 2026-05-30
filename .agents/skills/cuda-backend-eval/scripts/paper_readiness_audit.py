#!/usr/bin/env python3
"""Generate paper-readiness audit data from benchmark-viewer JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"
DEFAULT_RESULTS = VIEWER_DATA / "results.json"


def fail(message: str) -> None:
    raise SystemExit(f"paper readiness audit failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def require_records(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    records = data.get(key)
    if not isinstance(records, list):
        fail(f"{key} is missing or not a list")
    if not all(isinstance(record, dict) for record in records):
        fail(f"{key} contains a non-object record")
    return records


def result_index(results: dict[str, Any]) -> set[tuple[str, str, str]]:
    records = require_records(results, "result_records")
    index: set[tuple[str, str, str]] = set()
    for record in records:
        hardware = record.get("hardware")
        if not isinstance(hardware, dict):
            continue
        benchmark_id = record.get("benchmark_id")
        method_id = record.get("method_id")
        gpu = hardware.get("gpu")
        if all(isinstance(value, str) for value in (benchmark_id, method_id, gpu)):
            index.add((benchmark_id, method_id, gpu))
    return index


def count_evidence_refs(
    evidence_refs: list[dict[str, Any]],
    current_results: set[tuple[str, str, str]],
) -> tuple[dict[str, int], list[str]]:
    counts = Counter()
    missing_viewer_results: list[str] = []
    for ref in evidence_refs:
        kind = ref.get("kind")
        if not isinstance(kind, str):
            kind = "unknown"
        counts[kind] += 1
        if kind == "viewer_result":
            key = (
                str(ref.get("benchmark_id", "")),
                str(ref.get("method_id", "")),
                str(ref.get("gpu", "")),
            )
            if key not in current_results:
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
                }
            )
            continue
        statuses.append(
            {
                "paper_baseline_id": baseline_id,
                "latest_status": probe["latest_status"],
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


def claim_blockers(
    *,
    paper_baseline_ids: list[str],
    missing_evidence: list[str],
    missing_viewer_results: list[str],
    run_statuses: list[dict[str, Any]],
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
    return blockers


def build_readiness_audit(
    *,
    matrix: dict[str, Any],
    runs: dict[str, Any],
    probes: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    claims = require_records(matrix, "paper_evaluation_matrix")
    run_records = require_records(runs, "paper_baseline_runs")
    probe_records = require_records(probes, "paper_baseline_probes")
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
        blockers = claim_blockers(
            paper_baseline_ids=paper_baseline_ids,
            missing_evidence=missing_evidence,
            missing_viewer_results=missing_results,
            run_statuses=run_statuses,
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
                "missing_viewer_results": missing_results,
                "paper_baseline_run_statuses": run_statuses,
                "probe_statuses": probe_items,
                "blockers": blockers,
                "promotion_gate": claim["promotion_gate"],
            }
        )

    ready_claims = sum(1 for claim in claim_audits if claim["ready_for_paper_claim"])
    return {
        "schema_version": 1,
        "source_files": [
            "docs/nvidia-backend/benchmark-viewer/data/paper_evaluation_matrix.json",
            "docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json",
            "docs/nvidia-backend/benchmark-viewer/data/paper_baseline_probes.json",
            "docs/nvidia-backend/benchmark-viewer/data/results.json",
        ],
        "overall_status": "paper_ready"
        if ready_claims == len(claim_audits)
        else "not_paper_ready",
        "ready_claims": ready_claims,
        "blocked_claims": len(claim_audits) - ready_claims,
        "claim_audits": claim_audits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_readiness_audit(
        matrix=load_json(args.matrix),
        runs=load_json(args.runs),
        probes=load_json(args.probes),
        results=load_json(args.results),
    )
    if args.output:
        write_json(args.output, audit)
        print(f"wrote {repo_relative(args.output)}")
    else:
        print(json.dumps(audit, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
