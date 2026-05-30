#!/usr/bin/env python3
"""Update committed paper-baseline probe status from paired probe artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"
MACHINE_ARTIFACTS = {
    "A100": "a100-probe.json",
    "H200": "h200-probe.json",
}


def fail(message: str) -> None:
    raise SystemExit(f"paper probe status update failed: {message}")


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
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def aggregate_status(machine_statuses: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in machine_statuses}
    if statuses == {"pass"}:
        return "pass"
    if statuses == {"fail"}:
        return "fail"
    return "partial"


def load_machine_probe_index(
    paired_artifact_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    by_machine: dict[str, dict[str, dict[str, Any]]] = {}
    for gpu, filename in MACHINE_ARTIFACTS.items():
        artifact_path = paired_artifact_root / filename
        payload = load_json(artifact_path)
        probes = payload.get("probes")
        if not isinstance(probes, list):
            fail(f"{artifact_path} has no probes list")
        indexed: dict[str, dict[str, Any]] = {}
        for probe in probes:
            if not isinstance(probe, dict):
                fail(f"{artifact_path} contains non-object probe")
            baseline_id = probe.get("paper_baseline_id")
            if not isinstance(baseline_id, str) or not baseline_id:
                fail(f"{artifact_path} contains probe without paper_baseline_id")
            indexed[baseline_id] = probe
        by_machine[gpu] = indexed
    return by_machine


def update_probe_records(
    probe_data: dict[str, Any],
    *,
    paired_artifact_root: Path,
) -> dict[str, Any]:
    records = probe_data.get("paper_baseline_probes")
    if not isinstance(records, list):
        fail("paper_baseline_probes is missing or not a list")
    machine_index = load_machine_probe_index(paired_artifact_root)
    artifact_root_text = repo_relative(paired_artifact_root) + "/"

    updated = json.loads(json.dumps(probe_data))
    for record in updated["paper_baseline_probes"]:
        if not isinstance(record, dict):
            fail("paper_baseline_probes contains a non-object record")
        baseline_id = record.get("paper_baseline_id")
        if not isinstance(baseline_id, str) or not baseline_id:
            fail("probe record missing paper_baseline_id")
        machine_statuses: list[dict[str, Any]] = []
        for gpu, filename in MACHINE_ARTIFACTS.items():
            artifact_probe = machine_index[gpu].get(baseline_id)
            if artifact_probe is None:
                fail(f"{filename} missing baseline {baseline_id}")
            status = artifact_probe.get("status")
            if status not in {"pass", "partial", "fail", "not_captured"}:
                fail(f"{filename} has invalid status for {baseline_id}: {status}")
            blocking_gaps = artifact_probe.get("blocking_gaps", [])
            if not isinstance(blocking_gaps, list) or not all(
                isinstance(item, str) for item in blocking_gaps
            ):
                fail(f"{filename} has invalid blocking_gaps for {baseline_id}")
            machine_statuses.append(
                {
                    "gpu": gpu,
                    "status": status,
                    "artifact": artifact_root_text + filename,
                    "blocking_gaps": blocking_gaps,
                }
            )
        record["latest_artifact_root"] = artifact_root_text
        record["latest_machine_status"] = machine_statuses
        record["latest_status"] = aggregate_status(machine_statuses)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--paired-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updated = update_probe_records(
        load_json(args.probes),
        paired_artifact_root=args.paired_artifact_root,
    )
    write_json(args.output, updated)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
