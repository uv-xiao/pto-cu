"""Build isolated environment plan artifacts for paper baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_baseline_environment_plan_impl.dependency import dependency_evidence
from paper_baseline_environment_plan_impl.errors import fail
from paper_baseline_environment_plan_impl.paths import ROOT
from paper_baseline_environment_plan_impl.paths import repo_relative
from paper_baseline_environment_plan_impl.specs import ENVIRONMENT_SPECS


def baseline_records(baselines: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = baselines.get("paper_baselines")
    if not isinstance(records, list):
        fail("paper_baselines is missing or not a list")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            by_id[record["id"]] = record
    return by_id


def build_environment_plan(
    baseline_id: str,
    baseline: dict[str, Any],
    spec: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    source = baseline.get("source", {})
    source_path = str(source.get("local_tmp_path", ""))
    source_commit = str(source.get("commit", "unknown"))
    source_root = ROOT / source_path
    source_short = source_commit[:8] if len(source_commit) >= 8 else source_commit
    env_path = (
        ROOT
        / "tmp"
        / "cuda-backend"
        / "paper-baselines"
        / "envs"
        / f"{baseline_id}-{source_short}"
    )
    overlay_path = (
        ROOT
        / "tmp"
        / "cuda-backend"
        / "paper-baselines"
        / "source-overlays"
        / f"{baseline_id}-{source_short}-spinloop-cpython"
    )
    build_source_path = (
        repo_relative(overlay_path)
        if spec.get("source_overlay_steps")
        else source_path
    )
    env_python = f"{repo_relative(env_path)}/bin/python"
    env_bin = f"{repo_relative(env_path)}/bin"
    format_args = {
        "source_path": source_path,
        "build_source_path": build_source_path,
        "env_python": env_python,
        "env_bin": env_bin,
    }
    dependency_sources = list(spec["dependency_sources"])
    evidence = dependency_evidence(source_root, dependency_sources)
    critical_packages = []
    missing = []
    for package in spec["critical_packages"]:
        normalized = package.replace("_", "-").lower()
        package_evidence = evidence.get(normalized, [])
        if not package_evidence:
            missing.append(package)
        critical_packages.append(
            {
                "name": package,
                "declared": bool(package_evidence),
                "evidence": package_evidence,
            }
        )
    create_command = (
        f"python3 -m venv --system-site-packages {repo_relative(env_path)}"
    )
    install_commands = [create_command]
    install_commands.extend(
        step.format(**format_args)
        for step in spec["install_steps"]
    )
    source_overlay_commands = [
        step.format(**format_args) for step in spec.get("source_overlay_steps", [])
    ]
    install_commands.extend(source_overlay_commands)
    preflight_after_install_steps = len(install_commands)
    preflight_commands = [
        step.format(**format_args)
        for step in spec.get("preflight_steps", [])
    ]
    install_commands.extend(
        step.format(**format_args)
        for step in spec.get("install_after_preflight_steps", [])
    )
    validation_commands = [
        (
            "env PYTHONNOUSERSITE=1 "
            f"PYTHONPATH=$PWD/{build_source_path}:$PWD/{build_source_path}/python:$PYTHONPATH "
            f"{env_python} -c \"import importlib; "
            f"importlib.import_module('{module}')\""
        )
        for module in spec["validation_modules"]
    ]
    status = "plan_ready" if source_root.is_dir() and not missing else "partial"
    next_action = (
        "Run the install_commands on the evaluation host, then run the "
        "validation_commands before starting serving benchmarks."
    )
    return {
        "id": f"{baseline_id}_runtime_environment",
        "paper_baseline_id": baseline_id,
        "title": spec["title"],
        "status": status,
        "source_path": source_path,
        "source_commit": source_commit,
        "build_source_path": build_source_path,
        "source_overlay_commands": source_overlay_commands,
        "environment_path": repo_relative(env_path),
        "python_policy": (
            "Create a dedicated venv under tmp/ with --system-site-packages; "
            "never install these serving framework dependencies into the "
            "project .venv or user site."
        ),
        "dependency_sources": dependency_sources,
        "critical_packages": critical_packages,
        "manual_packages": list(spec.get("manual_packages", [])),
        "install_commands": install_commands,
        "preflight_commands": preflight_commands,
        "preflight_after_install_steps": preflight_after_install_steps,
        "validation_commands": validation_commands,
        "execution_gaps": [
            "Environment has not been materialized by this planner artifact.",
            "Serving benchmarks still need raw JSON capture after validation passes.",
            *[
                f"Critical package is not declared in inspected sources: {package}"
                for package in missing
            ],
        ],
        "notes": spec["notes"],
        "next_action": next_action,
        "raw_artifact": repo_relative(output_root / "environment-plans.json"),
    }


def build_environment_plans(
    *,
    baselines: dict[str, Any],
    output_root: Path,
    commit: str,
) -> dict[str, Any]:
    by_id = baseline_records(baselines)
    plans = []
    for baseline_id, spec in ENVIRONMENT_SPECS.items():
        baseline = by_id.get(baseline_id)
        if baseline is None:
            fail(f"missing paper baseline: {baseline_id}")
        plans.append(
            build_environment_plan(
                baseline_id,
                baseline,
                spec,
                output_root=output_root,
            )
        )
    return {
        "schema_version": 1,
        "metadata": {
            "pto_commit": commit,
            "artifact_root": repo_relative(output_root) + "/",
            "source_files": [
                "evaluations/nvidia/benchmark-viewer/data/paper_baselines.json",
            ],
        },
        "paper_baseline_environment_plans": plans,
    }
