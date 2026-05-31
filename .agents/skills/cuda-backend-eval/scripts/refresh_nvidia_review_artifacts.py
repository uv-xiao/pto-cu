#!/usr/bin/env python3
"""Regenerate NVIDIA review JSON artifacts in dependency order."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_ROOT = ROOT / ".agents" / "skills" / "cuda-backend-eval" / "scripts"
VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"


def fail(message: str) -> None:
    raise SystemExit(f"NVIDIA review artifact refresh failed: {message}")


def load_module(name: str) -> Any:
    path = SCRIPT_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VIEWER_DATA,
        help="Directory for generated JSON artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_module = load_module("paper_readiness_audit")
    work_queue_module = load_module("paper_readiness_work_queue")
    goal_progress_module = load_module("nvidia_goal_progress")

    matrix = audit_module.load_json(audit_module.DEFAULT_MATRIX)
    runs = audit_module.load_json(audit_module.DEFAULT_RUNS)
    probes = audit_module.load_json(audit_module.DEFAULT_PROBES)
    run_readiness = audit_module.load_json(audit_module.DEFAULT_RUN_READINESS)
    results = audit_module.load_json(audit_module.DEFAULT_RESULTS)
    audit = audit_module.build_readiness_audit(
        matrix=matrix,
        runs=runs,
        probes=probes,
        run_readiness=run_readiness,
        results=results,
    )
    audit_path = output_dir / "paper_readiness_audit.json"
    audit_module.write_json(audit_path, audit)
    print(f"wrote {audit_path}")

    work_queue = work_queue_module.build_work_queue(
        audit,
        audit_path=VIEWER_DATA / "paper_readiness_audit.json",
    )
    work_queue_path = output_dir / "paper_readiness_work_queue.json"
    work_queue_module.write_json(work_queue_path, work_queue)
    print(f"wrote {work_queue_path}")

    baselines = goal_progress_module.load_json(goal_progress_module.DEFAULT_BASELINES)
    goal_progress = goal_progress_module.build_goal_progress(
        audit=audit,
        work_queue=work_queue,
        matrix=matrix,
        baselines=baselines,
    )
    goal_progress_path = output_dir / "goal_progress.json"
    goal_progress_module.write_json(goal_progress_path, goal_progress)
    print(f"wrote {goal_progress_path}")


if __name__ == "__main__":
    main()
