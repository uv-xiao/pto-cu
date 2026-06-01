#!/usr/bin/env python3
"""Run and record bounded paper-baseline environment setup attempts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_data_io import load_json as load_viewer_json
from viewer_data_io import write_json as write_viewer_json

VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_PLANS = VIEWER_DATA / "paper_baseline_environment_plans.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "tmp" / "cuda-backend" / "paper-baselines" / "environment-attempts"
)
DEFAULT_VIEWER_OUTPUT = VIEWER_DATA / "paper_baseline_environment_attempts.json"


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline environment attempt failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def is_viewer_output(path: Path) -> bool:
    return (
        path.resolve().parent == VIEWER_DATA.resolve()
        or path.with_suffix("").is_dir()
    )


def load_viewer_output(path: Path) -> dict[str, Any]:
    if is_viewer_output(path):
        return load_viewer_json(path)
    return load_json(path)


def write_viewer_output(path: Path, payload: dict[str, Any]) -> None:
    if is_viewer_output(path):
        write_viewer_json(path, payload)
        return
    write_json(path, payload)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def plan_for_baseline(plans: dict[str, Any], baseline_id: str) -> dict[str, Any]:
    records = plans.get("paper_baseline_environment_plans")
    if not isinstance(records, list):
        fail("paper_baseline_environment_plans is missing or not a list")
    for record in records:
        if isinstance(record, dict) and record.get("paper_baseline_id") == baseline_id:
            return record
    fail(f"missing environment plan for baseline: {baseline_id}")


def command_is_allowed(command: str) -> bool:
    return ".venv" not in command and "--user" not in command


def run_step(
    *,
    command: str,
    index: int,
    kind: str,
    output_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not command_is_allowed(command):
        fail(f"refusing unsafe environment command: {command}")
    log_path = output_root / f"step-{index:02d}.log"
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.monotonic() - started
        output = result.stdout
        returncode: int | None = result.returncode
        status = "pass" if result.returncode == 0 else "fail"
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        returncode = None
        status = "timeout"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", errors="replace")
    return {
        "index": index,
        "kind": kind,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(duration, 3),
        "command": command,
        "log": repo_relative(log_path),
    }


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
    if max_steps <= 0:
        fail("--max-steps must be positive")
    if start_step <= 0:
        fail("--start-step must be positive")
    if timeout_seconds <= 0:
        fail("--timeout-seconds must be positive")

    plan = plan_for_baseline(plans, baseline_id)
    install_commands = list(plan.get("install_commands", []))
    preflight_commands = list(plan.get("preflight_commands", []))
    validation_commands = list(plan.get("validation_commands", []))
    if not install_commands:
        fail(f"{plan.get('id', baseline_id)} has no install_commands")
    if not validation_commands:
        fail(f"{plan.get('id', baseline_id)} has no validation_commands")
    preflight_after_install_steps = plan.get(
        "preflight_after_install_steps",
        len(install_commands),
    )
    if (
        not isinstance(preflight_after_install_steps, int)
        or isinstance(preflight_after_install_steps, bool)
        or preflight_after_install_steps < 0
        or preflight_after_install_steps > len(install_commands)
    ):
        fail(f"{plan.get('id', baseline_id)} has invalid preflight_after_install_steps")

    commands: list[tuple[str, str]] = [
        *[
            ("install", command)
            for command in install_commands[:preflight_after_install_steps]
        ],
        *[("preflight", command) for command in preflight_commands],
        *[
            ("install", command)
            for command in install_commands[preflight_after_install_steps:]
        ],
        *[("validation", command) for command in validation_commands],
    ]
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
            blocked = (
                f"step {index} {kind} {step['status']} for command: {command}"
            )
            break

    failed = any(step["status"] != "pass" for step in steps)
    steps_completed = len(steps)
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
    if status == "partial":
        next_action = (
            "Continue the remaining install and validation steps before serving "
            "benchmark execution."
        )
    elif status == "fail":
        next_action = (
            "Inspect the failed step log, resolve the recorded blocker, then "
            f"rerun this attempt with --start-step {start_step}."
        )
    else:
        next_action = (
            "Environment setup and validation passed; run the serving "
            "benchmark commands and import their raw JSON results."
        )
    attempt = {
        "id": (
            f"{baseline_id}_environment_attempt_"
            f"{commit.replace('-', '_')}{suffix}"
        ),
        "paper_baseline_id": baseline_id,
        "environment_plan_id": plan["id"],
        "title": f"{plan['title']} setup attempt",
        "status": status,
        "environment_path": plan["environment_path"],
        "artifact_root": repo_relative(output_root) + "/",
        "start_step": start_step,
        "end_step": end_step,
        "steps_completed": steps_completed,
        "steps_total": steps_total,
        "steps": steps,
        "artifacts": artifacts,
        "blocker": blocked,
        "observation": (
            "Bounded environment setup attempt captured command logs and JSON "
            "evidence under tmp/."
        ),
        "next_action": next_action,
    }
    payload = {
        "schema_version": 1,
        "metadata": {
            "pto_commit": commit,
            "artifact_root": repo_relative(output_root) + "/",
            "source_files": [
                "docs/nvidia-backend/benchmark-viewer/data/paper_baseline_environment_plans.json",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
    parser.add_argument("--baseline", required=True, choices=["vllm", "sglang"])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--viewer-output", type=Path, default=DEFAULT_VIEWER_OUTPUT)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--attempt-id-suffix", default="")
    parser.add_argument("--append-viewer", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.commit or git_commit()
    output_root = args.output_root
    if output_root is None:
        output_root = DEFAULT_OUTPUT_ROOT / f"{args.baseline}-{commit}"
    payload = build_attempt(
        plans=load_json(args.plans),
        baseline_id=args.baseline,
        output_root=output_root,
        commit=commit,
        max_steps=args.max_steps,
        start_step=args.start_step,
        timeout_seconds=args.timeout_seconds,
        attempt_id_suffix=args.attempt_id_suffix,
    )
    if args.append_viewer:
        payload = append_viewer_attempts(args.viewer_output, payload)
    write_viewer_output(args.viewer_output, payload)
    print(f"wrote {repo_relative(output_root / 'environment-attempt.json')}")


if __name__ == "__main__":
    main()
