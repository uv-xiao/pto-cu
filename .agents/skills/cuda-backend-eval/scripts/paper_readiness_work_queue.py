#!/usr/bin/env python3
"""Generate paper-readiness work-queue data from the readiness audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_data_io import load_json as load_viewer_json

VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_AUDIT = VIEWER_DATA / "paper_readiness_audit.json"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"


def fail(message: str) -> None:
    raise SystemExit(f"paper readiness work queue failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = load_viewer_json(path)
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    except ValueError as exc:
        fail(str(exc))
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


def require_string(record: dict[str, Any], key: str, owner: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        fail(f"{owner} has empty or missing {key}")
    return value


def build_work_queue(
    audit: dict[str, Any],
    *,
    runs: dict[str, Any] | None = None,
    audit_path: Path = DEFAULT_AUDIT,
) -> dict[str, Any]:
    claims = audit.get("claim_audits")
    if not isinstance(claims, list):
        fail("paper readiness audit has no claim_audits list")
    run_serving_workloads = baseline_run_serving_workloads(runs)

    work_items: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            fail("paper readiness audit contains a non-object claim")
        claim_id = require_string(claim, "id", "paper readiness audit claim")
        claim_title = require_string(claim, "title", claim_id)
        actions = claim.get("next_actions")
        if not isinstance(actions, list):
            fail(f"{claim_id} has no next_actions list")
        if claim.get("ready_for_paper_claim") is True and actions:
            fail(f"{claim_id} is ready but still has next actions")
        for action_index, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                fail(f"{claim_id} next action is not an object")
            source = require_string(action, "source", claim_id)
            paper_baseline_id = str(action.get("paper_baseline_id", ""))
            paper_baseline_run_id = str(action.get("paper_baseline_run_id", ""))
            execution_attempt_id = str(action.get("execution_attempt_id", ""))
            serving_workload_ids = action.get("serving_workload_ids", [])
            if not isinstance(serving_workload_ids, list) or not all(
                isinstance(item, str) for item in serving_workload_ids
            ):
                fail(f"{claim_id} next action has invalid serving_workload_ids")
            command_selectors = serving_command_plan_selectors(
                paper_baseline_run_id,
                serving_workload_ids,
                run_serving_workloads,
            )
            owner = paper_baseline_run_id or paper_baseline_id or source
            item_index = len(work_items) + 1
            work_items.append(
                {
                    "id": f"paper_readiness_work_item_{item_index:03d}",
                    "priority": item_index,
                    "claim_id": claim_id,
                    "claim_title": claim_title,
                    "matrix_status": require_string(
                        claim, "matrix_status", claim_id
                    ),
                    "ready_for_paper_claim": bool(
                        claim.get("ready_for_paper_claim", False)
                    ),
                    "blocker_count": len(claim.get("blockers", [])),
                    "missing_evidence_count": claim.get(
                        "missing_evidence_count", 0
                    ),
                    "action_index": action_index,
                    "source": source,
                    "owner": owner,
                    "paper_baseline_id": paper_baseline_id,
                    "paper_baseline_run_id": paper_baseline_run_id,
                    "execution_attempt_id": execution_attempt_id,
                    "missing_evidence_id": str(
                        action.get("missing_evidence_id", "")
                    ),
                    "method_id": str(action.get("method_id", "")),
                    "serving_workload_ids": serving_workload_ids,
                    "serving_command_plan_selectors": command_selectors,
                    "shape_contains": str(action.get("shape_contains", "")),
                    "status": require_string(action, "status", claim_id),
                    "action": require_string(action, "action", claim_id),
                    "evidence_summary": evidence_summary(action, claim_id),
                    "promotion_gate": require_string(
                        claim, "promotion_gate", claim_id
                    ),
                }
            )

    by_source = Counter(item["source"] for item in work_items)
    by_claim = Counter(item["claim_id"] for item in work_items)
    return {
        "schema_version": 1,
        "source_file": repo_relative(audit_path),
        "overall_status": require_string(audit, "overall_status", "audit"),
        "ready_claims": audit.get("ready_claims", 0),
        "blocked_claims": audit.get("blocked_claims", 0),
        "summary": {
            "total_work_items": len(work_items),
            "work_items_by_source": dict(sorted(by_source.items())),
            "work_items_by_claim": dict(sorted(by_claim.items())),
        },
        "work_items": work_items,
    }


def serving_command_plan_selectors(
    paper_baseline_run_id: str,
    serving_workload_ids: list[str],
    run_serving_workloads: dict[str, list[str]],
) -> list[str]:
    if not paper_baseline_run_id:
        return []
    allowed_serving_ids = run_serving_workloads.get(paper_baseline_run_id)
    if allowed_serving_ids is not None:
        allowed = set(allowed_serving_ids)
        serving_workload_ids = [
            serving_id
            for serving_id in serving_workload_ids
            if serving_id in allowed
        ]
    return [
        f"{paper_baseline_run_id}:{serving_workload_id}"
        for serving_workload_id in serving_workload_ids
    ]


def baseline_run_serving_workloads(
    runs: dict[str, Any] | None,
) -> dict[str, list[str]]:
    if runs is None:
        return {}
    records = runs.get("paper_baseline_runs", [])
    if not isinstance(records, list):
        fail("paper baseline runs has no paper_baseline_runs list")
    run_serving_workloads: dict[str, list[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            fail("paper baseline runs contains a non-object record")
        run_id = str(record.get("id", ""))
        serving_ids = record.get("serving_workload_ids", [])
        if not run_id:
            fail("paper baseline run has empty id")
        if not isinstance(serving_ids, list) or not all(
            isinstance(serving_id, str) for serving_id in serving_ids
        ):
            fail(f"{run_id} has invalid serving_workload_ids")
        run_serving_workloads[run_id] = serving_ids
    return run_serving_workloads


def evidence_summary(action: dict[str, Any], owner: str) -> list[str]:
    summary = action.get("evidence_summary", [])
    if summary in (None, ""):
        return []
    if not isinstance(summary, list) or not all(
        isinstance(item, str) and item.strip() for item in summary
    ):
        fail(f"{owner} action evidence_summary is invalid")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT,
        help="Input paper_readiness_audit.json path.",
    )
    parser.add_argument(
        "--baseline-runs",
        type=Path,
        default=DEFAULT_RUNS,
        help="Input paper_baseline_runs.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VIEWER_DATA / "paper_readiness_work_queue.json",
        help="Output work-queue JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = load_json(args.audit)
    runs = load_json(args.baseline_runs)
    payload = build_work_queue(audit, runs=runs, audit_path=args.audit)
    write_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
