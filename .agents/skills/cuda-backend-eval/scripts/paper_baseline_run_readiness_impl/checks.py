from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

from .paths import ROOT


def python_entrypoints(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    entrypoints: list[str] = []
    for idx, token in enumerate(tokens[:-1]):
        if token in {"python", "python3"} or token.endswith("/python"):
            candidate = tokens[idx + 1]
            if candidate.endswith(".py"):
                entrypoints.append(candidate)
    return entrypoints


def check_entrypoints(run: dict[str, Any], source_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    commands = run.get("run_commands", [])
    if not isinstance(commands, list):
        return checks
    for command in commands:
        if not isinstance(command, str):
            continue
        for entrypoint in python_entrypoints(command):
            if entrypoint in seen:
                continue
            seen.add(entrypoint)
            path = (
                ROOT / entrypoint
                if entrypoint.startswith(".agents/")
                else source_root / entrypoint
            )
            checks.append(
                {
                    "kind": "path_exists",
                    "path": entrypoint,
                    "status": "pass" if path.is_file() else "fail",
                    "why": "Python entrypoint referenced by run_commands.",
                }
            )
    return checks


def command_checks(run: dict[str, Any]) -> list[dict[str, str]]:
    commands = run.get("run_commands", [])
    status = (
        "pass"
        if commands and all(isinstance(item, str) for item in commands)
        else "fail"
    )
    return [
        {
            "kind": "run_command_contract",
            "status": status,
            "why": "Run readiness must be tied to explicit reproduction commands.",
        }
    ]


def artifact_checks(run: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    for artifact in run.get("expected_artifacts", []):
        if not isinstance(artifact, str):
            continue
        checks.append(
            {
                "kind": "expected_artifact_path",
                "path": artifact,
                "status": "pass" if artifact.startswith("tmp/") else "fail",
                "why": "Expected artifacts must remain under tmp/.",
            }
        )
    return checks


def probe_checks(
    baseline_id: str,
    probes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    probe = probes.get(baseline_id)
    if probe is None:
        return [
            {
                "kind": "baseline_probe",
                "name": baseline_id,
                "status": "partial",
                "why": (
                    "Each long paper-baseline run should reference a source "
                    "and dependency readiness probe."
                ),
                "blocking_gaps": [
                    f"No paper-baseline readiness probe exists for {baseline_id}."
                ],
            }
        ]

    machine_gaps: list[str] = []
    for machine in probe.get("latest_machine_status", []):
        if not isinstance(machine, dict):
            continue
        gpu = machine.get("gpu", "unknown")
        for gap in machine.get("blocking_gaps", []):
            if isinstance(gap, str):
                machine_gaps.append(f"{gpu}: {gap}")

    return [
        {
            "kind": "baseline_probe",
            "name": probe.get("id", baseline_id),
            "status": probe.get("latest_status", "partial"),
            "why": "Latest committed source/dependency probe status for this baseline.",
            "blocking_gaps": machine_gaps,
        }
    ]


def metric_checks(run: dict[str, Any]) -> list[dict[str, str]]:
    metrics = run.get("required_metrics", [])
    metric_set = set(metrics) if isinstance(metrics, list) else set()
    checks: list[dict[str, str]] = []
    for metric in ("correctness", "raw_artifacts"):
        checks.append(
            {
                "kind": "required_metric",
                "metric": metric,
                "status": "pass" if metric in metric_set else "fail",
                "why": (
                    "Paper-baseline imports must include correctness and raw "
                    "artifact evidence."
                ),
            }
        )
    return checks


def model_access_checks(run: dict[str, Any], baseline_id: str) -> list[dict[str, str]]:
    model_access = run.get("model_access")
    if isinstance(model_access, dict):
        requires_hf_token = model_access.get("requires_hf_token")
        if requires_hf_token is False:
            return [
                {
                    "kind": "environment",
                    "name": "public_model_access",
                    "status": "pass",
                    "why": str(
                        model_access.get(
                            "why",
                            "Selected model policy uses public model artifacts.",
                        )
                    ),
                }
            ]
        if requires_hf_token is True:
            return [
                {
                    "kind": "environment",
                    "name": "HF_TOKEN",
                    "status": "pass" if os.environ.get("HF_TOKEN") else "partial",
                    "why": str(
                        model_access.get(
                            "why",
                            "Selected model policy requires gated Hugging Face access.",
                        )
                    ),
                }
            ]

    if baseline_id in {"mpk", "vdcores"}:
        return [
            {
                "kind": "environment",
                "name": "HF_TOKEN",
                "status": "pass" if os.environ.get("HF_TOKEN") else "partial",
                "why": (
                    "Selected MPK/VDCores model commands require gated "
                    "Hugging Face model access."
                ),
            }
        ]
    return []


def environment_checks(
    run: dict[str, Any],
    baseline_id: str,
    source_root: Path,
    env_plans: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    checks = model_access_checks(run, baseline_id)
    plan = env_plans.get(baseline_id)
    if plan is not None and baseline_id in {"vllm", "sglang"}:
        checks.append(
            {
                "kind": "environment_plan",
                "name": str(plan.get("id", baseline_id)),
                "path": str(plan.get("raw_artifact", "")),
                "status": "pass" if plan.get("status") == "plan_ready" else "partial",
                "why": (
                    "Serving framework dependencies must be installed in an "
                    "isolated tmp/ environment before benchmark execution."
                ),
            }
        )
    if baseline_id == "vdcores":
        runtime_candidates = list((source_root / "python" / "dae").glob("runtime*.so"))
        checks.append(
            {
                "kind": "compiled_extension",
                "path": "python/dae/runtime*.so",
                "status": "pass" if runtime_candidates else "partial",
                "why": (
                    "VDCores run commands import dae.runtime from the built "
                    "CUDA extension."
                ),
            }
        )
    return checks


def blocking_gaps(checks: list[dict[str, Any]]) -> list[str]:
    gaps: list[str] = []
    for check in checks:
        if check["status"] == "pass":
            continue
        if check["kind"] == "environment":
            gaps.append(f"{check['name']} is not available for this run.")
        elif check["kind"] == "compiled_extension":
            gaps.append(f"Missing dae.runtime compiled extension at {check['path']}.")
        elif check["kind"] == "baseline_probe":
            detail = "; ".join(check.get("blocking_gaps", []))
            if detail:
                gaps.append(
                    f"Readiness probe {check['name']} is {check['status']}: {detail}."
                )
            else:
                gaps.append(f"Readiness probe {check['name']} is {check['status']}.")
        else:
            subject = check.get("path", check.get("metric", check["kind"]))
            gaps.append(f"{check['kind']} check failed for {subject}.")
    return gaps


def readiness_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        return "fail"
    if "partial" in statuses:
        return "partial"
    return "pass"
