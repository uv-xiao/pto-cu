from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import *  # noqa: F403
from .evidence import *  # noqa: F403


def validate_paper_baseline_environment_plans(
    data: dict[str, Any],
    baseline_ids: set[str],
    root: Path,
) -> None:
    if data.get("schema_version") != 1:
        fail("paper baseline environment plans schema_version must be 1")
    metadata = require_dict(data, "metadata", "paper baseline environment plans")
    artifact_root = require_string(
        metadata,
        "artifact_root",
        "paper baseline environment plan metadata",
    )
    require_current_artifact_path(root, artifact_root, "environment plan metadata")
    source_files = set(
        require_list(metadata, "source_files", "environment plan metadata")
    )
    if source_files != {
        "evaluations/nvidia/benchmark-viewer/data/paper_baselines.json",
    }:
        fail("paper baseline environment plan source_files are stale")
    records = require_list(
        data,
        "paper_baseline_environment_plans",
        "paper baseline environment plans",
    )
    check_unique_ids(records, "paper baseline environment plan")
    allowed_status = {"plan_ready", "partial", "materialized"}
    covered_baselines: set[str] = set()
    for record in records:
        owner = f"paper baseline environment plan {record['id']}"
        for key in (
            "title",
            "status",
            "source_path",
            "source_commit",
            "build_source_path",
            "environment_path",
            "python_policy",
            "next_action",
            "raw_artifact",
        ):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id not in baseline_ids:
            fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
        covered_baselines.add(baseline_id)
        if not record["environment_path"].startswith("tmp/"):
            fail(f"{owner} environment_path must be under tmp/")
        if record["build_source_path"] != record["source_path"]:
            if not record["build_source_path"].startswith("tmp/"):
                fail(f"{owner} build_source_path must be under tmp/")
            if "source-overlays" not in record["build_source_path"]:
                fail(f"{owner} build_source_path must identify a source overlay")
        require_current_artifact_path(root, record["raw_artifact"], owner)
        for key in (
            "dependency_sources",
            "critical_packages",
            "install_commands",
            "validation_commands",
            "execution_gaps",
            "notes",
        ):
            require_list(record, key, owner)
        if not isinstance(record.get("preflight_commands"), list):
            fail(f"{owner} preflight_commands is not a list")
        if not isinstance(record.get("source_overlay_commands"), list):
            fail(f"{owner} source_overlay_commands is not a list")
        if record["build_source_path"] != record["source_path"]:
            if not record["source_overlay_commands"]:
                fail(f"{owner} overlay build source has no source_overlay_commands")
            overlay_text = " ".join(record["source_overlay_commands"])
            if record["source_path"] not in overlay_text:
                fail(f"{owner} overlay command does not reference source_path")
            if record["build_source_path"] not in overlay_text:
                fail(f"{owner} overlay command does not reference build_source_path")
        preflight_after = record.get("preflight_after_install_steps")
        if (
            not isinstance(preflight_after, int)
            or isinstance(preflight_after, bool)
            or preflight_after < 0
            or preflight_after > len(record["install_commands"])
        ):
            fail(f"{owner} has invalid preflight_after_install_steps")
        if ".venv" in " ".join(record["install_commands"]):
            fail(f"{owner} must not install into the project .venv")
        if "--user" in " ".join(record["install_commands"]):
            fail(f"{owner} must not use pip --user")
        for command in record["install_commands"]:
            if "-m pip install" in command and "PYTHONNOUSERSITE=1" not in command:
                fail(f"{owner} pip install must disable user-site packages")
            if "-m pip install" in command and "PATH=" not in command:
                fail(f"{owner} pip install must put the env bin on PATH")
        for package in record["critical_packages"]:
            if not isinstance(package, dict):
                fail(f"{owner} critical package is not an object")
            require_string(package, "name", owner)
            if not isinstance(package.get("declared"), bool):
                fail(f"{owner} critical package declared is not boolean")
            evidence = package.get("evidence")
            if not isinstance(evidence, list):
                fail(f"{owner} critical package evidence is not a list")
            if package["declared"] and not evidence:
                fail(f"{owner} declared critical package has no evidence")
        manual_packages = record.get("manual_packages")
        if not isinstance(manual_packages, list):
            fail(f"{owner} manual_packages is not a list")
        for package in manual_packages:
            if not isinstance(package, dict):
                fail(f"{owner} manual package is not an object")
            require_string(package, "name", owner)
            require_string(package, "why", owner)
        validation_text = " ".join(record["validation_commands"])
        if "PYTHONNOUSERSITE=1" not in validation_text:
            fail(f"{owner} validation must disable user-site packages")
    required = {"vllm", "sglang"}
    if not required <= covered_baselines:
        missing = sorted(required - covered_baselines)
        fail(f"environment plans missing baseline coverage: {missing}")


def validate_paper_baseline_environment_attempts(
    data: dict[str, Any],
    baseline_ids: set[str],
    environment_plan_ids: set[str],
    root: Path,
) -> None:
    if data.get("schema_version") != 1:
        fail("paper baseline environment attempts schema_version must be 1")
    metadata = require_dict(data, "metadata", "paper baseline environment attempts")
    artifact_root = require_string(
        metadata,
        "artifact_root",
        "paper baseline environment attempt metadata",
    )
    require_current_artifact_path(root, artifact_root, "environment attempt metadata")
    source_files = set(
        require_list(metadata, "source_files", "environment attempt metadata")
    )
    if source_files != {
        "evaluations/nvidia/benchmark-viewer/data/paper_baseline_environment_plans.json",
    }:
        fail("paper baseline environment attempt source_files are stale")

    records = require_list(
        data,
        "paper_baseline_environment_attempts",
        "paper baseline environment attempts",
    )
    check_unique_ids(records, "paper baseline environment attempt")
    allowed_status = {"pass", "partial", "fail"}
    allowed_step_status = {"pass", "fail", "timeout"}
    allowed_step_kind = {"install", "preflight", "validation"}
    for record in records:
        owner = f"paper baseline environment attempt {record['id']}"
        for key in (
            "title",
            "status",
            "environment_path",
            "artifact_root",
            "observation",
            "next_action",
        ):
            require_string(record, key, owner)
        if record["status"] not in allowed_status:
            fail(f"{owner} has invalid status: {record['status']}")
        baseline_id = require_string(record, "paper_baseline_id", owner)
        if baseline_id not in baseline_ids:
            fail(f"{owner} references unknown paper_baseline_id: {baseline_id}")
        plan_id = require_string(record, "environment_plan_id", owner)
        if plan_id not in environment_plan_ids:
            fail(f"{owner} references unknown environment_plan_id: {plan_id}")
        if not record["environment_path"].startswith("tmp/"):
            fail(f"{owner} environment_path must be under tmp/")
        attempt_root = require_string(record, "artifact_root", owner)
        if not attempt_root.startswith("tmp/"):
            fail(f"{owner} artifact_root must be under tmp/: {attempt_root}")
        if not (root / attempt_root).is_dir():
            fail(f"{owner} artifact_root is missing: {attempt_root}")
        for key in ("start_step", "end_step", "steps_completed", "steps_total"):
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                fail(f"{owner} has invalid {key}")
            if key != "end_step" and value <= 0:
                fail(f"{owner} has invalid {key}")
        if record["end_step"] < record["start_step"] - 1:
            fail(f"{owner} end_step is before the attempted window")
        if record["end_step"] > record["steps_total"]:
            fail(f"{owner} end_step is past steps_total")
        if record["steps_completed"] > record["steps_total"]:
            fail(f"{owner} completed more steps than total")
        steps = require_list(record, "steps", owner)
        if len(steps) != record["steps_completed"]:
            fail(f"{owner} steps_completed does not match steps length")
        if steps and steps[0].get("index") != record["start_step"]:
            fail(f"{owner} first captured step does not match start_step")
        if steps and steps[-1].get("index") != record["end_step"]:
            fail(f"{owner} last captured step does not match end_step")
        blocker = record.get("blocker", "")
        if not isinstance(blocker, str):
            fail(f"{owner} blocker is not a string")
        if record["status"] != "pass" and not blocker:
            fail(f"{owner} is not pass but has no blocker")
        if record["status"] == "partial" and not (
            record["end_step"] < record["steps_total"]
        ):
            fail(f"{owner} partial status must leave remaining steps")
        for step in steps:
            if not isinstance(step, dict):
                fail(f"{owner} step is not an object")
            for key in ("kind", "status", "command", "log"):
                require_string(step, key, owner)
            if step["kind"] not in allowed_step_kind:
                fail(f"{owner} has invalid step kind: {step['kind']}")
            if step["status"] not in allowed_step_status:
                fail(f"{owner} has invalid step status: {step['status']}")
            command = step["command"]
            if ".venv" in command or "--user" in command:
                fail(f"{owner} step command escapes environment policy")
            if "-m pip install" in command and "PYTHONNOUSERSITE=1" not in command:
                fail(f"{owner} pip install step must disable user-site packages")
            if "-m pip install" in command and "PATH=" not in command:
                fail(f"{owner} pip install step must put the env bin on PATH")
            log = step["log"]
            if not log.startswith("tmp/") or not (root / log).is_file():
                fail(f"{owner} step log is missing: {log}")
            duration = step.get("duration_seconds")
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                fail(f"{owner} step duration is invalid")
        artifacts = require_list(record, "artifacts", owner)
        has_json = False
        for artifact in artifacts:
            if not isinstance(artifact, str) or not artifact.startswith("tmp/"):
                fail(f"{owner} artifact must be under tmp/: {artifact}")
            path = root / artifact
            if not path.is_file():
                fail(f"{owner} artifact is missing: {artifact}")
            has_json = has_json or path.suffix == ".json"
        if not has_json:
            fail(f"{owner} needs at least one JSON artifact")

