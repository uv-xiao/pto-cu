#!/usr/bin/env python3
"""Generate ultimate-goal progress data from current NVIDIA artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from viewer_data_io import load_json as load_viewer_json
from nvidia_goal_progress_contract import build_goal_progress as build_goal_progress_contract

VIEWER_DATA = ROOT / "docs" / "nvidia-backend" / "benchmark-viewer" / "data"
DEFAULT_AUDIT = VIEWER_DATA / "paper_readiness_audit.json"
DEFAULT_WORK_QUEUE = VIEWER_DATA / "paper_readiness_work_queue.json"
DEFAULT_MATRIX = VIEWER_DATA / "paper_evaluation_matrix.json"
DEFAULT_BASELINES = VIEWER_DATA / "paper_baselines.json"


def fail(message: str) -> None:
    raise SystemExit(f"nvidia goal progress failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = load_viewer_json(path)
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    except ValueError as exc:
        fail(str(exc))
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


def path_exists(path: str) -> bool:
    target = ROOT / path
    if target.exists():
        return True
    return (
        target.suffix == ".json"
        and target.with_suffix("").is_dir()
        and (target.with_suffix("") / "index.json").is_file()
    )


def remaining_gap_refs_from_status() -> list[str]:
    status = ROOT / "docs" / "nvidia-backend" / "status.md"
    try:
        text = status.read_text(encoding="utf-8")
        body = text.split("\n## Remaining Gaps\n", 1)[1].split("\n## ", 1)[0]
    except FileNotFoundError:
        fail(f"missing status file: {status}")
    except IndexError:
        fail("status.md has no Remaining Gaps section")
    refs: list[str] = []
    for line in body.splitlines():
        if not line.startswith("- [") or "](" not in line or ")" not in line:
            continue
        relpath = line.split("](", 1)[1].split(")", 1)[0]
        refs.append(f"docs/nvidia-backend/{relpath}")
    if not refs:
        fail("status.md has no remaining-gap links")
    return refs


def build_goal_progress(
    *,
    audit: dict[str, Any],
    work_queue: dict[str, Any],
    matrix: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    return build_goal_progress_contract(
        audit=audit,
        work_queue=work_queue,
        matrix=matrix,
        baselines=baselines,
        default_audit=DEFAULT_AUDIT,
        default_work_queue=DEFAULT_WORK_QUEUE,
        default_matrix=DEFAULT_MATRIX,
        default_baselines=DEFAULT_BASELINES,
        path_exists=path_exists,
        repo_relative=repo_relative,
        backend_gap_refs=remaining_gap_refs_from_status(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--work-queue", type=Path, default=DEFAULT_WORK_QUEUE)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument(
        "--output",
        type=Path,
        default=VIEWER_DATA / "goal_progress.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_goal_progress(
        audit=load_json(args.audit),
        work_queue=load_json(args.work_queue),
        matrix=load_json(args.matrix),
        baselines=load_json(args.baselines),
    )
    write_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
