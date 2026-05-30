#!/usr/bin/env python3
"""Collect safe readiness probes for paper baseline source checkouts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_BASELINES = VIEWER_DATA / "paper_baselines.json"
DEFAULT_PROBES = VIEWER_DATA / "paper_baseline_probes.json"


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline probe failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def run_command(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": result.returncode,
            "output": result.stdout[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "returncode": 124,
            "output": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
        }


def git_commit(path: Path) -> str:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=path)
    if result["returncode"] != 0:
        return "unknown"
    return str(result["output"]).strip()


def nvidia_smi() -> list[str]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if result["returncode"] != 0:
        return []
    return [line.strip() for line in result["output"].splitlines() if line.strip()]


def nvcc_version() -> str:
    result = run_command(["nvcc", "--version"])
    if result["returncode"] != 0:
        return "unavailable"
    for line in result["output"].splitlines():
        if "release" in line:
            return line.strip()
    return result["output"].splitlines()[-1].strip()


def check_path_exists(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    relpath = str(check["path"])
    path = source_root / relpath
    return {
        "kind": "path_exists",
        "path": relpath,
        "why": check["why"],
        "status": "pass" if path.exists() else "fail",
    }


def check_py_compile(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    relpath = str(check["path"])
    path = source_root / relpath
    if not path.is_file():
        return {
            "kind": "py_compile",
            "path": relpath,
            "why": check["why"],
            "status": "fail",
            "output": "file missing",
        }
    result = run_command([sys.executable, "-m", "py_compile", str(path)])
    return {
        "kind": "py_compile",
        "path": relpath,
        "why": check["why"],
        "status": "pass" if result["returncode"] == 0 else "fail",
        "returncode": result["returncode"],
        "output": result["output"],
    }


def check_python_module(check: dict[str, Any]) -> dict[str, Any]:
    module = str(check["module"])
    found = importlib.util.find_spec(module) is not None
    return {
        "kind": "python_module",
        "module": module,
        "why": check["why"],
        "status": "pass" if found else "fail",
    }


def check_python_import(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    module = str(check["module"])
    pythonpath = check.get("pythonpath")
    env = os.environ.copy()
    if pythonpath:
        path = source_root / str(pythonpath)
        env["PYTHONPATH"] = (
            f"{path}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(path)
        )
    result = subprocess.run(
        [sys.executable, "-c", f"import importlib; importlib.import_module({module!r})"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
        env=env,
    )
    return {
        "kind": "python_import",
        "module": module,
        "pythonpath": pythonpath,
        "why": check["why"],
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "output": result.stdout[-4000:],
    }


def collect_check(source_root: Path, check: dict[str, Any]) -> dict[str, Any]:
    kind = check.get("kind")
    if kind == "path_exists":
        return check_path_exists(source_root, check)
    if kind == "py_compile":
        return check_py_compile(source_root, check)
    if kind == "python_module":
        return check_python_module(check)
    if kind == "python_import":
        return check_python_import(source_root, check)
    fail(f"unknown probe check kind: {kind}")


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
    checks = [
        collect_check(source_root, check)
        for check in probe.get("checks", [])
    ]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        default="tmp/cuda-backend/paper-baselines/probes/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = collect_probe_artifact(
        load_json(args.baselines),
        load_json(args.probes),
        artifact_root=args.artifact_root,
    )
    write_json(args.output, artifact)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
