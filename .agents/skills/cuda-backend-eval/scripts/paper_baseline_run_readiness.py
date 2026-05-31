#!/usr/bin/env python3
"""Probe paper-baseline run readiness without executing long benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_BASELINES = VIEWER_DATA / "paper_baselines.json"
DEFAULT_RUNS = VIEWER_DATA / "paper_baseline_runs.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "tmp" / "cuda-backend" / "paper-baselines" / "run-readiness"
)
DEFAULT_VIEWER_OUTPUT = VIEWER_DATA / "paper_baseline_run_readiness.json"
TARGET_RUN_IDS = {
    "mpk_persistent_scheduler_trace",
    "vdcores_resource_policy_trace",
}


def fail(message: str) -> None:
    raise SystemExit(f"paper baseline run readiness failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"JSON root is not an object: {path}")
    return data


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
            path = source_root / entrypoint
            checks.append(
                {
                    "kind": "path_exists",
                    "path": entrypoint,
                    "status": "pass" if path.is_file() else "fail",
                    "why": "Python entrypoint referenced by run_commands.",
                }
            )
    return checks


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
                "why": "Paper-baseline imports must include correctness and raw artifact evidence.",
            }
        )
    return checks


def environment_checks(baseline_id: str, source_root: Path) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    if baseline_id in {"mpk", "vdcores"}:
        checks.append(
            {
                "kind": "environment",
                "name": "HF_TOKEN",
                "status": "pass" if os.environ.get("HF_TOKEN") else "partial",
                "why": "Selected MPK/VDCores model commands require gated Hugging Face model access.",
            }
        )
    if baseline_id == "vdcores":
        runtime_candidates = list((source_root / "python" / "dae").glob("runtime*.so"))
        checks.append(
            {
                "kind": "compiled_extension",
                "path": "python/dae/runtime*.so",
                "status": "pass" if runtime_candidates else "partial",
                "why": "VDCores run commands import dae.runtime from the built CUDA extension.",
            }
        )
    return checks


def blocking_gaps(checks: list[dict[str, str]]) -> list[str]:
    gaps: list[str] = []
    for check in checks:
        if check["status"] == "pass":
            continue
        if check["kind"] == "environment":
            gaps.append(f"{check['name']} is not available for this run.")
        elif check["kind"] == "compiled_extension":
            gaps.append(f"Missing dae.runtime compiled extension at {check['path']}.")
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


def build_run_readiness(
    *,
    baselines: dict[str, Any],
    runs: dict[str, Any],
    output_root: Path,
    commit: str,
) -> dict[str, Any]:
    sources = source_by_baseline(baselines)
    records: list[dict[str, Any]] = []
    for run in require_records(runs, "paper_baseline_runs"):
        run_id = run.get("id")
        if run_id not in TARGET_RUN_IDS:
            continue
        baseline_id = run.get("paper_baseline_id")
        if not isinstance(baseline_id, str):
            fail(f"{run_id} missing paper_baseline_id")
        source_root = sources.get(baseline_id)
        if source_root is None:
            fail(f"{run_id} references baseline without local_tmp_path")

        checks = [
            {
                "kind": "source_path",
                "path": repo_relative(source_root),
                "status": "pass" if source_root.is_dir() else "fail",
                "why": "Baseline source checkout used by run_commands.",
            },
            *check_entrypoints(run, source_root),
            *artifact_checks(run),
            *metric_checks(run),
            *environment_checks(baseline_id, source_root),
        ]
        gaps = blocking_gaps(checks)
        records.append(
            {
                "id": f"{run_id}_readiness",
                "paper_baseline_run_id": run_id,
                "paper_baseline_id": baseline_id,
                "title": f"{run.get('title', run_id)} Readiness",
                "latest_status": readiness_status(checks),
                "latest_artifact_root": repo_relative(output_root) + "/",
                "checks": checks,
                "blocking_gaps": gaps,
                "next_action": "Resolve blocking gaps, execute the run_commands, then import measured raw JSON through paper_baseline_results_update.py.",
            }
        )
    return {
        "schema_version": 1,
        "metadata": {
            "pto_commit": commit,
            "source_files": [
                "docs/nvidia-backend/benchmark-viewer/data/paper_baselines.json",
                "docs/nvidia-backend/benchmark-viewer/data/paper_baseline_runs.json",
            ],
        },
        "paper_baseline_run_readiness": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--viewer-output", type=Path, default=DEFAULT_VIEWER_OUTPUT)
    parser.add_argument("--commit", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = args.commit or git_commit()
    payload = build_run_readiness(
        baselines=load_json(args.baselines),
        runs=load_json(args.runs),
        output_root=args.output_root,
        commit=commit,
    )
    write_json(args.output_root / "run-readiness.json", payload)
    write_json(args.viewer_output, payload)
    print(f"wrote {repo_relative(args.output_root / 'run-readiness.json')}")


if __name__ == "__main__":
    main()
