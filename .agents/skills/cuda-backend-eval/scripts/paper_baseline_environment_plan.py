#!/usr/bin/env python3
"""Materialize isolated environment plans for paper serving baselines."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_BASELINES = VIEWER_DATA / "paper_baselines.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "tmp" / "cuda-backend" / "paper-baselines" / "environment-plans"
)
DEFAULT_VIEWER_OUTPUT = VIEWER_DATA / "paper_baseline_environment_plans.json"

PACKAGE_RE = re.compile(r"^([A-Za-z0-9_.-]+)")


ENVIRONMENT_SPECS: dict[str, dict[str, Any]] = {
    "vllm": {
        "title": "vLLM Isolated Runtime Environment",
        "dependency_sources": [
            "pyproject.toml",
            "requirements/build/cuda.txt",
            "requirements/common.txt",
            "requirements/cuda.txt",
        ],
        "critical_packages": [
            "cmake",
            "ninja",
            "setuptools-rust",
            "setuptools-scm",
            "torch",
            "torchvision",
            "pydantic",
            "cbor2",
            "flashinfer-python",
            "tilelang",
        ],
        "manual_packages": [
            {
                "name": "uvloop",
                "why": (
                    "The pinned api_server.py imports uvloop, but uvloop is "
                    "not declared in the inspected runtime requirement files."
                ),
            }
        ],
        "install_steps": [
            "env PYTHONNOUSERSITE=1 PATH={env_bin}:$PATH "
            "{env_python} -m pip install --upgrade pip setuptools wheel",
            "env PYTHONNOUSERSITE=1 PATH={env_bin}:$PATH "
            "{env_python} -m pip install uvloop",
            "REPO_ROOT=$PWD && cd {source_path} && "
            "env PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install "
            "-r requirements/common.txt -r requirements/cuda.txt",
            "REPO_ROOT=$PWD && cd {source_path} && "
            "env PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install "
            "-r requirements/build/cuda.txt",
            "REPO_ROOT=$PWD && cd {source_path} && "
            "env PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install --no-build-isolation -e .",
        ],
        "validation_modules": [
            "vllm",
            "vllm.entrypoints.cli.main",
            "vllm.entrypoints.openai.api_server",
            "vllm.engine.arg_utils",
        ],
        "notes": [
            "The vLLM source declares torch==2.11.0 while the project venv "
            "may carry a different CUDA/PyTorch stack, so the serving "
            "baseline must use a dedicated environment under tmp/.",
            "Editable installation is kept in the isolated environment because "
            "the installed vllm module and console script are required before "
            "server and throughput runs.",
        ],
    },
    "sglang": {
        "title": "SGLang Isolated Runtime Environment",
        "dependency_sources": [
            "python/pyproject.toml",
        ],
        "critical_packages": [
            "torch",
            "torchvision",
            "orjson",
            "uvloop",
            "flashinfer_python",
            "tilelang",
        ],
        "manual_packages": [],
        "install_steps": [
            "env PYTHONNOUSERSITE=1 PATH={env_bin}:$PATH "
            "{env_python} -m pip install --upgrade pip setuptools wheel",
            "REPO_ROOT=$PWD && cd {source_path} && "
            "env PYTHONNOUSERSITE=1 PATH=$REPO_ROOT/{env_bin}:$PATH "
            "$REPO_ROOT/{env_python} -m pip install --no-build-isolation -e \"python[all]\"",
        ],
        "validation_modules": [
            "sglang",
            "orjson",
            "torchvision",
            "sglang.bench_serving",
            "sglang.bench_offline_throughput",
            "sglang.bench_one_batch",
        ],
        "notes": [
            "SGLang imports orjson and torchvision during benchmark module "
            "initialization, so dependency validation must run with "
            "PYTHONNOUSERSITE=1 to avoid user-site leakage.",
            "The pinned source declares torch==2.11.0 and CUDA 13 packages; "
            "installing it into the project venv would make PTO tests "
            "non-reproducible.",
        ],
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline environment plan failed: {message}")


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


def package_name(requirement: str) -> str:
    text = requirement.strip()
    if not text or text.startswith("#") or text.startswith("-"):
        return ""
    text = text.split(";", 1)[0].strip()
    text = text.split("[", 1)[0].strip()
    match = PACKAGE_RE.match(text)
    return match.group(1).replace("_", "-").lower() if match else ""


def strip_inline_comment(line: str) -> str:
    in_quote = False
    quote = ""
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
        if char == "#" and not in_quote:
            return line[:index]
    return line


def fallback_toml_dependencies(path: Path) -> list[str]:
    dependencies: list[str] = []
    in_dependencies = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = strip_inline_comment(raw_line).strip()
        if line.startswith("dependencies") or line.startswith("requires"):
            in_dependencies = "[" in line
            if line.endswith("]"):
                in_dependencies = False
            continue
        if not in_dependencies:
            continue
        if line.startswith("]"):
            in_dependencies = False
            continue
        if not line:
            continue
        dependencies.append(line.strip(",").strip("'\""))
    return dependencies


def toml_dependencies(path: Path) -> list[str]:
    if tomllib is None:
        return fallback_toml_dependencies(path)
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    dependencies: list[str] = []
    project = payload.get("project", {})
    build_system = payload.get("build-system", {})
    for value in project.get("dependencies", []):
        if isinstance(value, str):
            dependencies.append(value)
    for value in build_system.get("requires", []):
        if isinstance(value, str):
            dependencies.append(value)
    return dependencies


def requirements_dependencies(path: Path) -> list[str]:
    dependencies: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = strip_inline_comment(raw_line).strip()
        if not line or line.startswith("-"):
            continue
        dependencies.append(line)
    return dependencies


def dependency_evidence(
    source_root: Path,
    dependency_sources: list[str],
) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    for source in dependency_sources:
        path = source_root / source
        if not path.is_file():
            continue
        dependencies = (
            toml_dependencies(path)
            if path.suffix == ".toml"
            else requirements_dependencies(path)
        )
        for dependency in dependencies:
            name = package_name(dependency)
            if not name:
                continue
            evidence.setdefault(name, []).append(f"{source}: {dependency}")
    return evidence


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
    env_python = f"{repo_relative(env_path)}/bin/python"
    env_bin = f"{repo_relative(env_path)}/bin"
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
        step.format(source_path=source_path, env_python=env_python, env_bin=env_bin)
        for step in spec["install_steps"]
    )
    validation_commands = [
        (
            "env PYTHONNOUSERSITE=1 "
            f"PYTHONPATH=$PWD/{source_path}:$PWD/{source_path}/python:$PYTHONPATH "
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
                "docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json",
            ],
        },
        "paper_baseline_environment_plans": plans,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--viewer-output", type=Path, default=DEFAULT_VIEWER_OUTPUT)
    parser.add_argument("--commit", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.commit or git_commit()
    output_root = args.output_root
    payload = build_environment_plans(
        baselines=load_json(args.baselines),
        output_root=output_root,
        commit=commit,
    )
    write_json(output_root / "environment-plans.json", payload)
    write_json(args.viewer_output, payload)
    print(f"wrote {repo_relative(output_root / 'environment-plans.json')}")


if __name__ == "__main__":
    main()
