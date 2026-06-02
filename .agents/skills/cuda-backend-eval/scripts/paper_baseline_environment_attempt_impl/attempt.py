"""Build and merge paper baseline environment attempt artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_baseline_environment_attempt_impl.errors import fail
from paper_baseline_environment_attempt_impl.io import load_viewer_output
from paper_baseline_environment_attempt_impl.io import write_json
from paper_baseline_environment_attempt_impl.paths import repo_relative
from paper_baseline_environment_attempt_impl.plans import plan_for_baseline
from paper_baseline_environment_attempt_impl.runner import run_step


def build_attempt(
    *,
    plans: dict[str, Any],
    baseline_id: str,
    output_root: Path,
    commit: str,
    max_steps: int,
    start_step: int,
    timeout_seconds: int,
    attempt_id_suffix: str,
) -> dict[str, Any]:
    _validate_attempt_args(max_steps, start_step, timeout_seconds)
    plan = plan_for_baseline(plans, baseline_id)
    commands = _environment_commands(plan, baseline_id)
    steps_total = len(commands)
    if start_step > steps_total:
        fail(f"--start-step {start_step} is past the {steps_total} planned steps")

    selected_commands = commands[start_step - 1 : start_step - 1 + max_steps]
    output_root.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    blocked = ""
    for index, (kind, command) in enumerate(selected_commands, start=start_step):
        step = run_step(
            command=command,
            index=index,
            kind=kind,
            output_root=output_root,
            timeout_seconds=timeout_seconds,
        )
        steps.append(step)
        if step["status"] != "pass":
            blocked = f"step {index} {kind} {step['status']} for command: {command}"
            break

    attempt = _attempt_record(
        baseline_id=baseline_id,
        commit=commit,
        attempt_id_suffix=attempt_id_suffix,
        plan=plan,
        output_root=output_root,
        start_step=start_step,
        steps_total=steps_total,
        steps=steps,
        blocked=blocked,
    )
    payload = {
        "schema_version": 1,
        "metadata": {
            "pto_commit": commit,
            "artifact_root": repo_relative(output_root) + "/",
            "source_files": [
                "evaluations/nvidia/benchmark-viewer/data/paper_baseline_environment_plans.json",
            ],
        },
        "paper_baseline_environment_attempts": [attempt],
    }
    write_json(output_root / "environment-attempt.json", payload)
    return payload


def append_viewer_attempts(viewer_output: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if not viewer_output.is_file() and not viewer_output.with_suffix("").is_dir():
        return payload
    existing = load_viewer_output(viewer_output)
    existing_records = existing.get("paper_baseline_environment_attempts")
    new_records = payload.get("paper_baseline_environment_attempts")
    if not isinstance(existing_records, list) or not isinstance(new_records, list):
        fail(f"invalid environment attempts JSON: {viewer_output}")
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in [*existing_records, *new_records]:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            fail(f"invalid environment attempt record in {viewer_output}")
        identifier = record["id"]
        if identifier not in by_id:
            order.append(identifier)
        by_id[identifier] = record
    merged = dict(payload)
    merged["paper_baseline_environment_attempts"] = [
        by_id[identifier] for identifier in order
    ]
    return merged


def _validate_attempt_args(
    max_steps: int,
    start_step: int,
    timeout_seconds: int,
) -> None:
    if max_steps <= 0:
        fail("--max-steps must be positive")
    if start_step <= 0:
        fail("--start-step must be positive")
    if timeout_seconds <= 0:
        fail("--timeout-seconds must be positive")


def _environment_commands(
    plan: dict[str, Any],
    baseline_id: str,
) -> list[tuple[str, str]]:
    install_commands = list(plan.get("install_commands", []))
    preflight_commands = list(plan.get("preflight_commands", []))
    validation_commands = list(plan.get("validation_commands", []))
    if not install_commands:
        fail(f"{plan.get('id', baseline_id)} has no install_commands")
    if not validation_commands:
        fail(f"{plan.get('id', baseline_id)} has no validation_commands")
    split = plan.get("preflight_after_install_steps", len(install_commands))
    if (
        not isinstance(split, int)
        or isinstance(split, bool)
        or split < 0
        or split > len(install_commands)
    ):
        fail(f"{plan.get('id', baseline_id)} has invalid preflight_after_install_steps")
    return [
        *[("install", command) for command in install_commands[:split]],
        *[("preflight", command) for command in preflight_commands],
        *[("install", command) for command in install_commands[split:]],
        *[("validation", command) for command in validation_commands],
    ]


def _attempt_record(
    *,
    baseline_id: str,
    commit: str,
    attempt_id_suffix: str,
    plan: dict[str, Any],
    output_root: Path,
    start_step: int,
    steps_total: int,
    steps: list[dict[str, Any]],
    blocked: str,
) -> dict[str, Any]:
    failed = any(step["status"] != "pass" for step in steps)
    end_step = steps[-1]["index"] if steps else start_step - 1
    status = "fail" if failed else "pass"
    if not failed and end_step < steps_total:
        status = "partial"
        blocked = (
            f"bounded attempt stopped at step {end_step} of {steps_total} "
            "environment steps"
        )
    artifacts = [step["log"] for step in steps]
    artifacts.append(repo_relative(output_root / "environment-attempt.json"))
    suffix = f"_{attempt_id_suffix}" if attempt_id_suffix else ""
    return {
        "id": f"{baseline_id}_environment_attempt_{commit.replace('-', '_')}{suffix}",
        "paper_baseline_id": baseline_id,
        "environment_plan_id": plan["id"],
        "title": f"{plan['title']} setup attempt",
        "status": status,
        "environment_path": plan["environment_path"],
        "artifact_root": repo_relative(output_root) + "/",
        "start_step": start_step,
        "end_step": end_step,
        "steps_completed": len(steps),
        "steps_total": steps_total,
        "steps": steps,
        "artifacts": artifacts,
        "blocker": blocked,
        "observation": (
            "Bounded environment setup attempt captured command logs and JSON "
            "evidence under tmp/."
        ),
        "next_action": _next_action(status, start_step),
    }


def _next_action(status: str, start_step: int) -> str:
    if status == "partial":
        return (
            "Continue the remaining install and validation steps before serving "
            "benchmark execution."
        )
    if status == "fail":
        return (
            "Inspect the failed step log, resolve the recorded blocker, then "
            f"rerun this attempt with --start-step {start_step}."
        )
    return (
        "Environment setup and validation passed; run the serving benchmark "
        "commands and import their raw JSON results."
    )
