from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403


def validate_paper_baseline_probes(
    data: dict[str, Any],
    baseline_ids: set[str],
    root: Path,
) -> None:
    records = require_list(data, "paper_baseline_probes", "paper baseline probes")
    check_unique_ids(records, "paper baseline probe")
    allowed_status = {"not_captured", "pass", "partial", "fail"}
    allowed_kinds = {
        "path_exists",
        "py_compile",
        "python_import",
        "python_module",
    }
    covered_baselines: set[str] = set()
    for record in records:
        owner = f"paper baseline probe {record['id']}"
        for key in ("title", "latest_status", "latest_artifact_root", "next_action"):
            require_string(record, key, owner)
        if record["latest_status"] not in allowed_status:
            fail(f"{owner} has invalid latest_status: {record['latest_status']}")
        if not record["latest_artifact_root"].startswith("tmp/"):
            fail(f"{owner} latest_artifact_root must be under tmp/")
        require_current_artifact_path(root, record["latest_artifact_root"], owner)
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id not in baseline_ids:
            fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
        covered_baselines.add(baseline_id)
        checks = require_list(record, "checks", owner)
        machine_status = require_list(record, "latest_machine_status", owner)
        machine_gpus: set[str] = set()
        machine_statuses: set[str] = set()
        for status_record in machine_status:
            if not isinstance(status_record, dict):
                fail(f"{owner} latest_machine_status entry is not an object")
            gpu = require_string(status_record, "gpu", owner)
            if gpu not in {"A100", "H200"}:
                fail(f"{owner} has invalid machine status GPU: {gpu}")
            if gpu in machine_gpus:
                fail(f"{owner} has duplicate machine status for {gpu}")
            machine_gpus.add(gpu)
            status = require_string(status_record, "status", owner)
            if status not in allowed_status:
                fail(f"{owner} has invalid machine status: {status}")
            machine_statuses.add(status)
            artifact = require_string(status_record, "artifact", owner)
            artifact_payload = load_current_json_artifact(
                root,
                artifact,
                owner,
            )
            gaps = status_record.get("blocking_gaps", [])
            if not isinstance(gaps, list):
                fail(f"{owner} machine blocking_gaps is not a list")
            for gap in gaps:
                if not isinstance(gap, str) or not gap:
                    fail(f"{owner} has invalid machine blocking gap")
            artifact_probes = {
                item.get("paper_baseline_id"): item
                for item in artifact_payload.get("probes", [])
                if isinstance(item, dict)
            }
            artifact_probe = artifact_probes.get(baseline_id)
            if not artifact_probe:
                fail(f"{owner} artifact {artifact} missing baseline {baseline_id}")
            if artifact_probe.get("status") != status:
                fail(
                    f"{owner} machine status for {gpu} does not match "
                    f"{artifact}: {status} != {artifact_probe.get('status')}"
                )
            if artifact_probe.get("blocking_gaps", []) != gaps:
                fail(
                    f"{owner} blocking gaps for {gpu} do not match "
                    f"{artifact}"
                )
        if {"A100", "H200"} != machine_gpus:
            fail(f"{owner} must include A100 and H200 machine status")
        if record["latest_status"] == "pass" and machine_statuses != {"pass"}:
            fail(f"{owner} latest_status pass disagrees with machine statuses")
        if record["latest_status"] == "partial" and "partial" not in machine_statuses:
            fail(f"{owner} latest_status partial disagrees with machine statuses")
        modules: set[str] = set()
        for check in checks:
            if not isinstance(check, dict):
                fail(f"{owner} check is not an object")
            kind = require_string(check, "kind", owner)
            if kind not in allowed_kinds:
                fail(f"{owner} has invalid check kind: {kind}")
            require_string(check, "why", owner)
            if kind in {"path_exists", "py_compile"}:
                require_string(check, "path", owner)
            if kind in {"python_import", "python_module"}:
                modules.add(require_string(check, "module", owner))
            if kind == "python_import" and "pythonpath" in check:
                require_string(check, "pythonpath", owner)
        if baseline_id == "thunderkittens":
            required_modules = {
                "torch",
                "pybind11",
                "numpy",
                "pandas",
                "matplotlib",
                "tqdm",
            }
            if not required_modules <= modules:
                missing = sorted(required_modules - modules)
                fail(f"{owner} missing ThunderKittens modules: {missing}")
        if baseline_id in {"mpk", "vdcores"} and "transformers" not in modules:
            fail(f"{owner} missing Transformers module probe")
    required_baselines = {"mpk", "vdcores", "vllm", "sglang", "thunderkittens"}
    if not required_baselines <= covered_baselines:
        missing = sorted(required_baselines - covered_baselines)
        fail(f"paper baseline probes missing baseline coverage: {missing}")


