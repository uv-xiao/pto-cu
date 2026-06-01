from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import fail
from .paths import ROOT


def require_records(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    records = data.get(key)
    if not isinstance(records, list):
        fail(f"{key} is missing or not a list")
    if not all(isinstance(record, dict) for record in records):
        fail(f"{key} contains a non-object record")
    return records


def source_by_baseline(baselines: dict[str, Any]) -> dict[str, Path]:
    by_id: dict[str, Path] = {}
    for baseline in require_records(baselines, "paper_baselines"):
        baseline_id = baseline.get("id")
        source = baseline.get("source")
        if not isinstance(baseline_id, str) or not isinstance(source, dict):
            continue
        local_path = source.get("local_tmp_path")
        if isinstance(local_path, str) and local_path:
            by_id[baseline_id] = (ROOT / local_path).resolve()
    return by_id


def probe_by_baseline(probes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for probe in require_records(probes, "paper_baseline_probes"):
        baseline_id = probe.get("paper_baseline_id")
        if isinstance(baseline_id, str):
            by_id[baseline_id] = probe
    return by_id


def environment_plan_by_baseline(
    env_plans: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for plan in env_plans.get("paper_baseline_environment_plans", []):
        if not isinstance(plan, dict):
            continue
        baseline_id = plan.get("paper_baseline_id")
        if isinstance(baseline_id, str):
            by_id[baseline_id] = plan
    return by_id
