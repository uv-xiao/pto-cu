from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any

from .checks import collect_check
from .commands import git_commit
from .commands import nvcc_version
from .commands import nvidia_smi
from .errors import fail
from .io import repo_path
from .paths import ROOT


def probe_status(commit_matches: bool, checks: list[dict[str, Any]]) -> str:
    statuses = [check["status"] for check in checks]
    if commit_matches and all(status == "pass" for status in statuses):
        return "pass"
    if any(status == "pass" for status in statuses):
        return "partial"
    return "fail"


def collect_probe(
    probe: dict[str, Any],
    baselines_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_id = probe["paper_baseline_id"]
    baseline = baselines_by_id.get(baseline_id)
    if baseline is None:
        fail(f"probe {probe['id']} references unknown baseline {baseline_id}")
    source = baseline["source"]
    source_root = repo_path(source["local_tmp_path"])
    expected_commit = source["commit"]
    actual_commit = git_commit(source_root) if source_root.is_dir() else "missing"
    checks = [collect_check(source_root, check) for check in probe.get("checks", [])]
    commit_matches = actual_commit == expected_commit
    blocking_gaps = []
    if not source_root.is_dir():
        blocking_gaps.append("source checkout missing")
    if not commit_matches:
        blocking_gaps.append("source commit mismatch")
    blocking_gaps.extend(
        f"{check['kind']} failed: {check.get('path', check.get('module'))}"
        for check in checks
        if check["status"] != "pass"
    )
    return {
        "probe_id": probe["id"],
        "paper_baseline_id": baseline_id,
        "title": probe["title"],
        "status": probe_status(commit_matches, checks),
        "source_path": source["local_tmp_path"],
        "source_commit_expected": expected_commit,
        "source_commit_actual": actual_commit,
        "checks": checks,
        "blocking_gaps": blocking_gaps,
        "next_action": probe["next_action"],
    }


def collect_probe_artifact(
    baselines: dict[str, Any],
    probes: dict[str, Any],
    *,
    artifact_root: str,
) -> dict[str, Any]:
    baselines_by_id = {
        baseline["id"]: baseline for baseline in baselines["paper_baselines"]
    }
    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pto_commit": git_commit(ROOT),
            "artifact_root": artifact_root,
            "python": sys.version.split()[0],
            "nvcc": nvcc_version(),
            "gpus": nvidia_smi(),
        },
        "probes": [
            collect_probe(probe, baselines_by_id)
            for probe in probes["paper_baseline_probes"]
        ],
    }
